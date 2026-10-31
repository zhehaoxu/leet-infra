# Reference Approach: Write the Priority Queue as an Explicit Greedy Pass

## Implementation

1. Get the candidate set from the first $k$ columns of `torch.sort(probs, dim=-1, descending=True, stable=True)`. `stable=True` keeps "smaller expert index first" when probabilities tie, which is exactly the convention the statement fixes. `torch.topk` usually gives the same answer on small CPU tensors, but its tie order is not part of a contract, so the statement constrains the result, not the API.
2. `key = sorted_probs[:, :k] + jitter.gather(1, cand)` and `order = torch.argsort(-key.reshape(-1), stable=True)`. Negating the key and sorting stably is the same as ascending `(-key, s)`.
3. Do the claiming in a Python loop: `remaining` is a per-expert slack list, `used` is a per-token set of already claimed experts, `for slot in order.tolist(): token, rank = divmod(slot, k)`, then scan `cand[token][rank:]` for the first expert with slack that the token has not used.
4. Weights: `kept = assign >= 0`, `kept_probs = torch.where(kept, probs.gather(1, assign.clamp(min=0)), 0.0)`, `denom = kept_probs.sum(1, keepdim=True)`, `weights = torch.where(denom > 0, kept_probs / denom, 0.0)`.
5. `counts = torch.bincount(assign[kept], minlength=E)[:E]` and `dropped = T*k - int(kept.sum())`.

Four common mistakes:

- **Letting jitter into the top-k.** `topk(probs + jitter)` changes the candidate set itself; that is the explicit-noise gating convention, not this one, where jitter only orders the queue.
- **Processing in slot order.** Everything still lines up shape-wise while `assign` is completely different — capacity dispatch lives entirely in the processing order.
- **Feeding `assign == -1` straight into `gather`.** Either it raises, or you clamp to 0 and forget the `where`, and every dropped slot then picks up expert 0's probability and inflates the renormalization denominator.
- **Forgetting "one token never claims an expert twice".** When a low-rank slot is processed first, a high-rank slot of the same token may already have taken an expert; without the `used` check both slots pile onto it and `counts` goes wrong with them.

## Correctness checks

The 8 `check_*` functions cover: the hand example (all five returned items), randomized differential cases (8 `T/e/k/capacity/jitter_amp` combinations plus three noncontiguous views), dispatch invariants (`order` is a permutation whose keys are non-increasing, `counts <= capacity`, `counts.sum() == Tk - dropped`, distinct surviving experts per token, every assigned expert comes from the candidate set at a rank not earlier than its own, surviving weight rows sum to 1 or are all zero), no-contention and full-drop (`capacity=100` must collapse to plain top-k plus renorm; `capacity=0` must give all `-1`), jitter reordering and spill (same probabilities with and without jitter move the survivor from rank 0 to rank 1; a negative jitter flattens the keys to check the slot-id tie-break), degenerates and ties (all-equal probabilities, all-zero rows, a single slot, `k=E`, plus two all-integer-key 64-slot cases), invalid inputs, and the maximum shape.

The oracle is pure Python: `sorted(range(E), key=lambda e: (-p[t][e], e))[:k]` for candidates, `sorted(range(T*k), key=lambda s: (-keys[s//k][s%k], s))` for the queue, lists for `remaining`/`used`, and `math.fsum` for the renormalization. No torch `sort`/`argsort`/`topk`/`gather`/`bincount`.

Mutation testing catches: jitter inside the top-k, slot-order processing, `argsort` without `stable`, a spill chain that wraps back to rank 0, removing the "already used by this token" guard, not zeroing dropped positions, renormalizing over candidates instead of survivors, counting candidates instead of assignments into `counts`, dividing by an unguarded denominator, and treating "rank changed" as "dropped". The unstable `argsort` only shows up on the 64-slot heavy-tie case, which is what makes that case a necessary pin rather than padding. For E ≤ 8 an unstable in-row `torch.sort` is indistinguishable from a stable one on this machine — the statement pins the tie convention so that implementers know they depend on it, not because every violation is measurable.

## Scope and sources

**What has a source**: priority-ordered capacity dispatch and "renormalize only after dropping" can both be read in `NllbMoeTop2Router.route_tokens` in `transformers/models/nllb_moe/modeling_nllb_moe.py` (NLLB-MoE is the official implementation of the GShard idea). The relevant lines, with the configuration defaults `batch_prioritized_routing=False` and `normalize_router_prob_before_dropping=False`:

```text
# sort tokens based on their routing probability
# to make sure important tokens are routed, first
importance_scores = -1 * router_probs.max(dim=1)[0]
sorted_top_1_mask = top_1_mask[importance_scores.argsort(dim=0)]
sorted_cumsum1 = (torch.cumsum(sorted_top_1_mask, dim=0) - 1) * sorted_top_1_mask
locations1 = sorted_cumsum1[importance_scores.argsort(dim=0).argsort(dim=0)]
locations2 = sorted_cumsum2[...]
locations2 += torch.sum(top_1_mask, dim=0, keepdim=True)
top_1_mask = top_1_mask * torch.lt(locations1, self.expert_capacity)
top_2_mask = top_2_mask * torch.lt(locations2, self.expert_capacity)
top_1_max_probs = (router_probs * top_1_mask).sum(dim=1)
denom_s = torch.clamp(top_1_max_probs + top_2_max_probs, min=torch.finfo(router_probs.dtype).eps)
```

How they map: upstream queues tokens by descending routing probability before computing positions, which is isomorphic to this problem's descending-`key` greedy pass; `locations2 += sum(top_1_mask)` pushes the second choice's position behind the tokens that already took that expert as their first choice, which is the arithmetic of spilling over; the masks are capacity-filtered **before** being used as the renormalization denominator, which is this problem's "renormalize over survivors"; `clamp(min=eps)` is an alternative spelling of the zero-denominator rule, where this problem tests explicitly instead of introducing an arbitrary epsilon.

**What has no first-hand source**: perturbing the scores with uniform noise before dispatch (usually called jitter) is widely reported in the community, but the exact formula could not be checked against the original text here, so three conventions — `key = probs + jitter`, spill forward along the rank only, and no expert claimed twice by a token — are **defined by the statement** and carry no attribution. Making jitter an input tensor rather than an internal RNG draw is what keeps the judge reproducible.

Division of labour inside the collection: `moe-capacity-dispatch` is the minimal "drop by token position" policy, this problem is "queue by jittered priority and spill"; in `moe-noisy-topk-gating` the noise **does** change the top-k, here it does not (upstream's `second_expert_policy="sampling"` Gumbel noise is likewise added before the choice); the index algebra over candidates, weights and experts is in `moe-grouped-execution-plan`; the cross-step load accounting is in `moe-dead-expert-ledger`.
