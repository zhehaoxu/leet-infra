# Notes: FastV Visual Token Pruning at Layer K

## Approach

Validate everything first, then average `attention[:, :, -1, image_start:image_start + image_len]` over the head axis to get a per-row score ledger. Compute the keep count with Python `round(image_len * (1 - r))` -- not `int()` truncation and not numpy/torch rounding -- and clamp it into `[0, image_len]`.

Selection uses a stable descending sort: one `descending=True, stable=True` sort leaves tied scores in their original ascending index order, the first `k` entries are taken, and those indices are sorted ascending again before adding `image_start`. Prefix and suffix come from `torch.arange` expanded over the batch, and the three parts concatenate into `keep_indexs`.

The kept embeddings are collected along the token axis with `gather`, and `position_ids` is simply a copy of `keep_indexs`: the holes here are preserved original indices, not a compacted renumbering. `topk` is never called and no input is modified.

## Deliberate deviations from the official code

The upstream pruning step is one short block, but it carries four sources of engineering nondeterminism. This exercise pins each down as a rule instead of silently reproducing them:

- **Order.** After the official `torch.topk`, surviving visual tokens sit in `keep_indexs` in DESCENDING score order. This task re-emits them in ASCENDING index order so tokens and position ids stay causal, and the `hidden_states` gather follows that ascending order.
- **Ties.** `torch.topk` breaks equal scores in a backend-undefined way. Here "score descending, then index ascending" is explicit, so a tie keeps the earlier token.
- **Batching.** The upstream code indexes `[0]` and is therefore single-sequence. The same rule is applied row by row over a batch: every row selects its own visual tokens while sharing one `k`, so output lengths agree and both `keep_indexs` and `position_ids` carry a batch axis.
- **Scoring definition.** The paper text describes "average attention received from all tokens over all layers", while the official code uses only the LAST ROW of layer K averaged over heads. This task implements the CODE and requires non-final rows to be irrelevant; which layer is K is expressed by the attention matrix the caller passes in.

Two further observables are exercise additions rather than upstream behaviour: returning the per-token `score` ledger makes judging possible, and returning both `keep_indexs` and `position_ids` exposes implementations that secretly renumber.

## Independent validation

The oracle does not reuse the reference's tensor sorts or `mean`. It calls `tolist()` on both inputs, accumulates the final row's columns head by head with a scalar for loop, divides by the head count, selects with `sorted(range, key=lambda i: (-score[i], i))`, restores ascending index order with a second `sorted`, and rebuilds the kept embedding rows through Python list comprehensions before converting back to tensors. Randomised cases compare element by element against this oracle, and integer outputs must match exactly with `torch.equal`.

The 16 `check_*` functions cover the two executed examples, a half-to-even boundary table (`round(1.5)=2`, `round(2.5)=2`, `round(0.5)=0`), the `r=0` identity, full pruning at `k=0`, prefix/suffix survival under a large `r`, ties pinned to the lower index, gaps that are never renumbered, the fact that only the final row and the head mean matter, byte-exact copies of the original kept rows, ascending re-emission that differs from score order, float32 versus float64, noncontiguous views, input immutability, ten randomised shape sweeps, and every invalid input (`r` out of range or not real, bool masquerading as int, non-square attention, zero heads, dtype mismatch, non-finite values, and more).

## Complexity

Let `L' = L - (image_len - k)`. Scoring costs `O(B H image_len)`, sorting `O(B image_len log image_len)`, and gathering embeddings `O(B L' D)`, for a total of `O(B (H image_len + image_len log image_len + L' D))`. Apart from the outputs, only `O(B image_len)` scores and `O(B L')` index intermediates are needed. Continuing the attention over all `L` tokens after layer K instead costs the computation for the `image_len - k` removed tokens, which is exactly the saving FastV buys.

## Source and scope

Based on the already retrieved FastV repository `pkunlp-icler/FastV`, file `src/FastV/llava-hf/transformers/src/transformers/models/llama/modeling_llama.py`, the `FASTV_prune and layer_idx == FASTV_k` branch. Head averaging, the last-row read, `round(image_token_length * (1 - r))`, pruning only inside the visual span, the `keep_indexs` concatenation, and the `position_ids = keep_indexs.unsqueeze(0)` gap semantics all come from that code. The README defaults are `K=3, r=0.75` while the paper's headline configuration is `K=2, R=50%`; this exercise binds neither default and takes `r` and the visual span position entirely from arguments.

There is no LM head, KV cache, model forward, or HF `generate` mask adjustment, and the task does not decide which layer is K. The attention matrix must already be post-softmax, which is the caller's responsibility. This is a deterministic reimplementation of one pruning index step, not the complete FastV inference system.

```text
.venv-backend/bin/python scripts/add_problem.py --selftest mm-fastv-visual-token-prune --mode submit
```
