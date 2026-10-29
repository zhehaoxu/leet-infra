# Multimodal: Image Placeholder Expansion and Batch Assembly

Each text sequence contains one valid image placeholder. Remove invalid positions, replace that placeholder with all visual embeddings for the sample, and right-pad the resulting variable-length sequences into a batch.

This task restricts the original LLaVA placeholder-expansion subroutine to **one image per row, right padding, and no truncation**, and adds `mm_token_type_ids`. Embeddings are already provided; no model weights or vocabulary lookup are needed. This is not equivalent to the modern HF path where the processor first expands the placeholder into `N` image markers and the model then fills those positions with `masked_scatter`. Here, **one** placeholder is expanded into multiple positions by this operation itself.

## Inputs and bounds

All tensors are on CPU and may be noncontiguous.

| Input | Shape, type, and meaning |
| --- | --- |
| `text_embeds` | `float32` or `float64`, `[B,L,D]`; input embeddings including placeholder and padding positions |
| `labels` | `int64`, `[B,L]`; per-position labels |
| `attention_mask` | `bool`, `[B,L]`; `True` means valid; left padding, right padding, and internal holes are allowed |
| `image_mask` | `bool`, `[B,L]`; `True` marks an image placeholder |
| `image_features` | A length-`B` list of tensors; item `b` has shape `[N_b,D]` and the same dtype/device as `text_embeds` |

Bounds: `1 <= B <= 4`, `1 <= L <= 16`, `1 <= D <= 8`, and `1 <= N_b <= 12`.

Each row must contain **exactly one** image marker at a valid position. Raise `ValueError` for an incorrect marker count or for any `image_mask=True` position whose `attention_mask` is `False`. All other inputs are valid. Do not mutate any input tensor or the image-feature list.

## Expansion rules

Process each row independently:

1. Keep positions with `attention_mask=True` in their original order, discarding all others. Valid positions need not be contiguous.
2. Replace the single image placeholder with all `N_b` vectors in `image_features[b]`, in order. Discard the placeholder's own embedding and label.
3. Preserve every other valid text embedding and label unchanged. Set every new visual position's label to `-100`. **Do not shift labels or truncate.**
4. Number valid `position_ids` consecutively from `0`. Set `mm_token_type_ids` to `1` for visual positions and `0` for text positions.

A text label of `-100` only means that text position is ignored by the loss; it does not identify an image. Visual identity comes only from placeholder expansion.

## Return values

Let `A_b` be the number of valid input positions in row `b`. Its expanded length is `S_b = A_b - 1 + N_b`, and the batch length is `S = max_b S_b`, so `1 <= S <= 27`.

Return a dictionary with **exactly** these five keys. All outputs are on CPU, and each row is right-padded to `S`:

| Key | Shape and dtype | Valid positions | Right padding |
| --- | --- | --- | --- |
| `inputs_embeds` | `[B,S,D]`, input dtype | Expanded embeddings | `0` |
| `labels` | `[B,S]`, `int64` | Original text labels; `-100` for visual positions | `-100` |
| `attention_mask` | `[B,S]`, `bool` | `True` | `False` |
| `position_ids` | `[B,S]`, `int64` | `0,1,...,S_b-1` | `0` |
| `mm_token_type_ids` | `[B,S]`, `int64` | `1` for visual positions; `0` for text | `0` |

## Example

`text_embeds` and `image_features` use `float32`, `labels` uses `int64`, and both input masks use `bool`:

```text
text_embeds = [
  [[1,10], [99,99], [2,20], [3,30]],
  [[88,88], [4,40], [77,77], [5,50]]
]
labels = [[11,91,-100,13], [81,14,92,15]]
attention_mask = [
  [True, True, True, True],
  [False, True, False, True]
]
image_mask = [
  [False, True, False, False],
  [False, False, False, True]
]
image_features = [
  [[101,102], [103,104]],
  [[201,202]]
]
```

Output:

```text
inputs_embeds = [
  [[1,10], [101,102], [103,104], [2,20], [3,30]],
  [[4,40], [201,202], [0,0], [0,0], [0,0]]
]
labels = [
  [11,-100,-100,-100,13],
  [14,-100,-100,-100,-100]
]
attention_mask = [
  [True, True, True, True, True],
  [True, True, False, False, False]
]
position_ids = [[0,1,2,3,4], [0,1,0,0,0]]
mm_token_type_ids = [[0,1,1,0,0], [0,1,0,0,0]]
```

The expanded lengths are `5` and `2`. In the first row, `[2,20]` retains label `-100`, but its type id is `0` because it is text. The second row loses its left padding and internal hole, then receives three right-padding positions. This example is implemented and execution-checked by `check_example`.
