# Multimodal: FastV Visual Token Pruning at Layer K

## Task

At layer K of a vision-language model, score the visual tokens with the **last query row** of that layer's attention matrix, keep only the highest-scoring visual tokens, and delete the rest. The text prefix and the text after the image always survive, and surviving tokens continue to use their **original position ids**.

This extracts the single pruning step of FastV (Chen et al., arXiv:2403.06764, ECCV 2024 Oral) from the official implementation. It is **not the full inference pipeline**: the caller supplies the attention matrix, no model forward runs here, and attention is never recomputed. The bilingual starter contract is the single reference for parameters, shapes, numeric limits, and exceptions. The rules below define the computation.

## Scoring rules

Write the visual span as the absolute index range `[image_start, image_start + image_len)`; the last query row has index `L - 1`.

1. For every batch row `b`, the score of the `i`-th visual token is
   $$
   \mathrm{score}[b,i]=\frac{1}{H}\sum_{h=0}^{H-1}\mathrm{attention}[b,h,L-1,\mathrm{image\_start}+i].
   $$
   Average over heads first, then read the final row. Only that row is used, so every other query row must be irrelevant to the result.
2. The keep count is `k = round(image_len * (1 - r))`, where `r` is the **drop fraction** and `round` is Python's half-to-even rounding, so `round(1.5)` and `round(2.5)` both give `2`. `k` may be `0`, which removes the whole visual span.
3. Only tokens inside the visual span compete. The system-plus-question prefix before `image_start` and the trailing text after `image_start + image_len` are never pruned, no matter how large `r` is.
4. When taking the top `k` scores, ties resolve as "score descending, then index ascending", and the selected visual indices must then be **re-emitted in ascending index order** rather than in score order, otherwise the sequence is no longer causal.
5. The emitted index set is "prefix + ascending kept visual indices + suffix" with length `L' = L - (image_len - k)`. `position_ids` equals that set: holes in the numbering are **correct behaviour**, and renumbering from `0` is the common mistake.

The starter contract fixes the order, shapes, and dtypes of the four return values. Every pruned row is a byte-exact copy of an original row; no renormalization or rescaling is applied.

## Example 1: head averaging, ties, and gaps

All floating tensors use float64, `image_start = 2`, `image_len = 4`, `r = 0.5`, hence `k = round(4 * 0.5) = 2`. Only the two heads' final-row slices over the visual span are shown; all other rows and columns are irrelevant.

```text
hidden_states = [[[1, 10], [2, 20], [3, 30], [4, 40],
                  [5, 50], [6, 60], [7, 70], [8, 80]]]
image_start = 2, image_len = 4, r = 0.5
attention[0, :, 7, 2:6] = [[0.4, 0.2, 0.8, 0.6],
                           [0.6, 0.2, 0.2, 0.4]]

score                = [[0.5, 0.2, 0.5, 0.5]]
pruned_hidden_states = [[[1, 10], [2, 20], [3, 30],
                         [5, 50], [7, 70], [8, 80]]]
keep_indexs          = [[0, 1, 2, 4, 6, 7]]
position_ids         = [[0, 1, 2, 4, 6, 7]]
```

The four visual scores are `0.5, 0.2, 0.5, 0.5`; among the three-way tie the ascending rule keeps absolute indices `2` and `4` and drops `5`. The prefix `0, 1` and suffix `6, 7` stay untouched, and the numbering has holes exactly at the deleted `3` and `5`.

## Example 2: per-row selection and half-to-even

All floating tensors use float64, `B = 2`, one head, `image_start = 1`, `image_len = 3`, `r = 0.5`, hence `k = round(3 * 0.5) = round(1.5) = 2`.

```text
hidden_states = [[[1], [2], [3], [4], [5]],
                 [[10], [20], [30], [40], [50]]]
image_start = 1, image_len = 3, r = 0.5
attention[:, 0, 4, :] = [[0.2, 0.9, 0.1, 0.5, 0.3],
                         [0.2, 0.3, 0.7, 0.4, 0.1]]

score                = [[0.9, 0.1, 0.5], [0.3, 0.7, 0.4]]
pruned_hidden_states = [[[1], [2], [4], [5]],
                        [[10], [30], [40], [50]]]
keep_indexs          = [[0, 1, 3, 4], [0, 2, 3, 4]]
position_ids         = [[0, 1, 3, 4], [0, 2, 3, 4]]
```

Both rows use the same `k`, yet each row picks its own visual tokens: the first row deletes absolute index `2`, the second deletes `1`.

All displayed outputs were dumped from actual reference execution and checked against an independent oracle; the shown decimal values are exact.
