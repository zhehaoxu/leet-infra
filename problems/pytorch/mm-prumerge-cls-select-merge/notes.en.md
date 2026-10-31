# Analysis: LLaVA-PruMerge CLS-Attention Selection and Cosine Residual Merge

## Approach

Validate every argument first: `reduction_ratio` must be a Python real number in `(0,1]`, `eps` must be a positive real, and `int(N * reduction_ratio)` equal to `0` raises `ValueError` immediately, because "keep no token at all" is undefined for this task.

Selection is one sort with `descending=True` and `stable=True`, so tied attention values keep their original ascending indices; after taking the first `k` we sort once more to get ascending `kept_indexs`. `pruned_indexs` comes from a boolean mask plus `argsort(..., stable=True)`: the unselected `0` entries sort first and stay in ascending order, and slicing the first `N-k` columns yields exactly the merged-away patches. Both index tensors are ascending, disjoint, and together cover all of `0..N-1`.

The denominator used by `F.normalize(p=2, dim=-1, eps=eps)` is `max(||x||, eps)`, which is identical to the fallback promised by the contract, so a zero vector normalises to all zeros rather than NaN. Similarity uses `bmm`, the cluster head uses `argmax` (which automatically takes the first maximum on ties), probabilities use `softmax` across the whole kept axis, and we then `gather` the chosen column and multiply by `C ** -0.5`. The merge starts from a copy of `kept` and accumulates `weight * pruned` rows into the cluster column with `scatter_add`: that is precisely "own feature + every weighted neighbour assigned to it", and columns with no assignment keep their original value. When `N-k = 0` every intermediate is empty and `merged` equals the input patches reordered ascending.

## Deliberate deviations from the official implementation

The snippet in the official `llava/model/multimodal_encoder/clip_encoder.py` is the basis for these rules; every deviation below is pinned on purpose, not an incidental simplification:

- **Order**. `torch.topk` emits kept indices in attention **descending** order and the official snippet continues on that `idx` directly. This problem re-sorts the kept indices **ascending**, so `merged[j]` and `kept_indexs[j]` line up in the same order and the judge depends on one deterministic ordering only.
- **Ties**. The winner among equal values is backend-dependent for both `topk` and `argmax`. Here it is specified explicitly: selection breaks ties by "value descending, index ascending", cluster selection by "first maximum"; in the second example the all-zero cosine row of the zero-norm patch therefore lands on column 0.
- **Batch axis on index outputs**. In the official snippet the kept set is computed per row (the index list differs row by row), and this problem additionally returns `kept_indexs`, `pruned_indexs`, `cluster`, `weights` and `scores` with a batch axis `[B,k]` / `[B,N-k]`. These are new observable quantities introduced for the exercise, not part of the upstream return signature.
- **`int()` truncation kept**. The official `int(N * reduction_ratio)` truncates instead of rounding; this problem implements exactly that and states it in the contract so it is not confused with FastV's `round()`.
- **Softmax-then-scale kept**. The official code is `F.softmax(attn, dim=-1) * C ** -0.5`: probabilities are normalised over the **whole kept set** first, the cluster entry is picked afterwards, and the reciprocal square root of the channel count is applied last; there is no renormalisation inside a cluster. This counter-intuitive semantics is preserved in full and a test asserts that one cluster's weights do not sum to `1`.
- **eps fallback**. The official code relies on the default `eps` of `F.normalize`; this problem exposes `eps` as an argument and defines `norm_j = max(||x||_2, eps)`, which makes the zero-norm patch behaviour decidable.

## Independent validation

The oracle reuses none of torch's selection or reduction. It converts inputs with `tolist()` into nested lists, computes dot products and norms with `math.sqrt` and scalar loops, applies the `max(norm, eps)` fallback, hand-writes `argmax` as a loop with a strict `>` comparison (so ties take the first index), hand-writes a stable softmax by subtracting the maximum, using `math.exp` and summing, and then accumulates the residual per cluster in Python lists. Random cases are compared element-wise against this oracle, with the three index tensors and the cluster assignment required to match exactly via `torch.equal`.

The 16 `check_*` functions cover the two executed examples, the truncation-not-rounding boundary table (`int(5*0.5)=2`, `int(4*0.25)=1`, `int(3*0.5)=1`, `int(3*0.99)=2`), all-tied attention, `reduction_ratio=1` giving an empty pruned set with `merged` byte-identical to the input, a two-member cluster whose weights do not sum to `1`, weights equal to `p * C**-0.5` for `C=1/4/9`, a zero-norm patch free of NaN with identical results across several `eps` values, `argmax` ties taking the first index, empty clusters returned verbatim, the residual equalling "own plus assigned", `scores` equalling the maximum cosine and being invariant to patch magnitude scaling, float32/float64, noncontiguous views, input immutability, ten seeded random shape sweeps, and every invalid input (ratio out of range or not a real number, `k` truncating to `0`, `eps<=0`, shape mismatches, empty batch / empty patch set / zero feature dimension, dtype disagreement, non-finite values).

## Complexity

Let `k = int(N * reduction_ratio)`. Selection costs `O(B N log N)`, the cosine similarities `O(B (N-k) k C)`, softmax plus gather `O(B (N-k) k)`, and the residual accumulation `O(B N C)`, for a total of `O(B (N log N + (N-k) k C + N C))`. Apart from the outputs, the significant intermediates are the `(B, N-k, k)` similarity matrix and the `(B, N, C)` unit vectors, i.e. feature-copy-sized at `O(B (N-k) k C)`. With `k` around `N/8` the similarity matrix is far smaller than the returned feature copy.

## Source and scope

The source is the verified official LLaVA-PruMerge repository `42Shawn/LLaVA-PruMerge`, file `llava/model/multimodal_encoder/clip_encoder.py`: the CLS-row-over-patch-columns attention, the selection by `torch.topk(cls_attn, int(N * reduction_ratio), dim=1, largest=True)`, the complement `[c for c in range(N) if c not in row_idx]`, the post-`F.normalize` `bmm` cosine, the cluster head from `torch.argmax(cos_sim_matrix, dim=2)`, the weights `F.softmax(attn, dim=-1) * C ** -0.5`, and the residual merge `updated_center = x_others[b, i, :] + weighted_avg` all come from that implementation; the visual token reduction really does happen inside the vision tower, before the projector, keeping roughly `1/8` of the patches by default.

Deliberately out of scope: the paper decides how many tokens to keep per image adaptively with an "interquartile range + IQR outlier rule", while this problem fixes the count to the single parameter `int(N * reduction_ratio)`; the per-layer LLM-side pruning variant; the pooled output that concatenates CLS with the final token; and the real CLIP forward pass - the attention matrix and the patch features are explicit inputs and the caller guarantees that `cls_attn` really is the head-averaged CLS-to-patch attention. This is a deterministic reimplementation of the "select + merge" pair of steps, not the complete LLaVA-PruMerge model.

```text
.venv-backend/bin/python scripts/add_problem.py --selftest mm-prumerge-cls-select-merge --mode submit
```
