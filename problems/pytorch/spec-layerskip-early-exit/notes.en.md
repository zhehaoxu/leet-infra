# LayerSkip Early-Exit Scheduling: Editorial

## Source

LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding, Elhoushi, Shrivastava et al., FAIR at Meta, 2024, [arXiv:2404.16710](https://arxiv.org/abs/2404.16710), code [facebookresearch/LayerSkip](https://github.com/facebookresearch/LayerSkip). The passages that this problem is built on:

- §2.1/§4.1: training applies layer dropout (higher rates for later layers) plus an early-exit loss with **one shared LM head**, turning a single checkpoint into an ensemble of sub-models of different depths.
- §4.2 early-exit inference: "we run the first $E$ transformer layers in a model, and skip to the model's LM head" — unembed directly from layer $E$.
- §4.3.1 Self-Drafting: autoregressive drafting through the first $E$ layers; the draft length is `num_speculations` (our $\gamma$).
- §4.3.2 Self-Verification: "the self-verification stage critically only requires computing the remaining layers of the model that were not used in the draft stage. For a model with $L$ layers, the number of verification layers is $L-E$".
- §4.3.3 Reusing the Cache: a single KV cache shared by draft and verify (they use the same layers in the same order) plus an **exit query cache** holding only the query of layer $E-1$, jointly the **KVQ cache** — the physical reason verification can simply resume at layer $E$.
- §A.4 pseudocode: `verified_tokens = model.forward_remainder(input + draft_tokens, kvq_cache)` then `matched_tokens = match(draft_tokens, verified_tokens)` — acceptance is **draft token equals the full model's argmax**, not rejection sampling and not typical acceptance.

Two deliberate generalizations are stated in the problem itself: (1) the paper fixes one $E$ per model, while here the exit depth is **adaptive per draft position** (a confidence threshold `tau`, capped at $D$) — that is the per-token version of the paper's Figure 4 observation that "most of the time, the final token prediction is predicted fewer layers before the end"; (2) the paper only says verification costs $L-E$ layers, but here the $\gamma$ positions have different depths and a single batched forward can only reuse the cache prefix **every** position owns, so the cost is $\max(L-\min_j e_j, 1)$, the `1` being the LM-head pass that still has to produce the closing token.

## Core intuition: turn "number of layers" into accountable currency

Speculative decoding profits from "cheap compute does more steps, expensive compute does fewer". LayerSkip's trick is that the draft model and the verified model are **the same model**: the draft is the shallow prefix, verification is the deep prefix, weights and caches shared. The round's cost then decomposes exactly in transformer blocks:

- draft: $\gamma$ sequential shallow passes, position $j$ running $e_j$ blocks;
- verify: one parallel forward running $L-\min_j e_j$ blocks (all positions are aligned to the shallowest cached prefix and pushed from there to layer $L$);
- yield: the agreeing draft prefix of length $a$ plus one token that the full model produces.

The round pays off whenever $\sum_j e_j + (L - \min_j e_j) < (a+1)\,L$, which is exactly the intuition behind `speedup = sum(toks)·L/sum(lays) > 1`.

## Why the parallel forward aligns to min(e_j)

If the $\gamma$ draft positions exited at depths $2,3,1$, their cached "layers already computed" sets are $\{1,2\},\{1,2,3\},\{1\}$. A batched forward executes **one common set of layers** for all positions, so to reach layer $L$ it has to start at the first layer that even the shallowest position has not seen: $\min_j e_j + 1 = 2$, i.e. layers $2..5$ = 4 blocks when $L=5$. Deeper positions therefore **recompute** layers they already cached — the inherent price of batching (a batch must be homogeneous), which is what "shared cache prefix" means here. Using $\max_j e_j$ instead silently drops that recomputation, the most common slip in this problem.

## Cost accounting and the speedup definition

`lays` is **per request**, but `speedup` is a **single batch number**: $\text{sum(toks)}\cdot L/\text{sum(lays)}$, read as "how many blocks plain token-by-token full-model decoding would have needed for this batch, divided by how many were actually used". Because the denominator is a sum, token-rich requests dominate it; averaging the per-request ratios instead is noticeably optimistic in a mixed batch where one request sails through and another dies at position 0 (`check_speedup_is_token_weighted_not_mean_of_requests` pins exactly this with the extremes 2.4 and 0.667 against the true 1.4545).

## Numerical and implementation pitfalls

1. **Thresholds must be binary-exact.** In float32 `0.7` is really 0.699999988…, while the python float `0.7` is 0.69999999999999996, so a comparison that should tie does not. Every probability in the worked example is a multiple of $1/16$ (0.3125/0.5/0.625/0.75/0.8125) so that `tau=0.625` sitting exactly on the boundary is reproducible on the judge, not just on paper.
2. **Argmax ties take the smallest index.** `torch.argmax` does not document tie behaviour; scan explicitly (`if row[v] > m`, strict, left to right). Both the draft token and the target token must use the same rule, otherwise the agreement test gives false positives/negatives.
3. **Verification is never refunded.** If position 0 already disagrees the round yields a single token, but the $\gamma$ draft passes and the verification pass are still paid for — the most common cause of `speedup < 1`. Do not break out of the layer accounting early.
4. **The $\max(\cdot,1)$ floor.** When $D = L$ and every position runs all $L$ blocks, the remaining-layer count is 0, yet the closing token still needs the LM head, counted as 1 block; writing $L-\min e$ yields absurd values such as $\texttt{sum(lays)} \le 0$.
5. **Shape for $\gamma = 0$.** `depth` must be $(R,0)$ (one empty row per request), not $(0,)$; that case has `toks=1`, `lays=L` and speedup exactly 1, which makes it a perfect sentinel test.
6. **"First qualifying layer", not "shallowest qualifying layer"**: they agree, but if you scan from $D-1$ downwards (the oracle's trick) you must overwrite repeatedly with smaller depths, or the semantics flip.

## Cross-check methodology

`solution.py` scans layers forward, accumulates depths, uses `sum(es)` for the draft cost, `min(es)` for the verification cost and a sequential scan for the agreement length. The independent oracle in `test_cases.py` deliberately goes the other way everywhere: a **reverse** layer scan (`for l in reversed(range(D))`) overwriting with smaller depths; `max(range(V), key=...)` for the first-index argmax (Python's `max` returns the first maximal element); materialization of the executed block ids as an explicit `list(range(1, e+1))` concatenation whose `len()` *is* the layer count — that path validates the cost model itself as "number of blocks executed"; the first disagreement found by materializing a boolean list and taking its first `False`; batch totals via `math.fsum`. Random cases sweep $R,\gamma,D,V,L$ across 4 `tau` levels, plus 8 degenerate/invariant probes ($\gamma=0$ sentinel, `tau=0`, `tau>1`, all-equal rows exercising ties and the on-threshold boundary, total disagreement, target distribution cannot change the schedule, raising `tau` only deepens exits, the minimal $\gamma=1$ round).

## Complexity

The reference is $O(R\,\gamma\,(D + V))$ time and $O(R\,\gamma)$ extra space (plain python loops after `.tolist()`, no tensor allocation). The oracle materializes block ids, so it is $O(R(\gamma D + L))$ — still milliseconds. Unlike problem 283, both sides here are polynomial simulations, so the cross-check can be scaled to a few hundred random cases without eating the 30 s budget.
