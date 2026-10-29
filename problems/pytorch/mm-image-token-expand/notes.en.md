# Notes: Image Placeholder Expansion and Batch Assembly

## Source and restrictions

The source is LLaVA's [`llava/model/llava_arch.py`](https://raw.githubusercontent.com/haotian-liu/LLaVA/main/llava/model/llava_arch.py), specifically the padding removal, placeholder expansion, and batch assembly in lines 212–324 of `prepare_inputs_labels_for_multimodal`. The link tracks `main`; line numbers refer to the verified version and may move later.

This is a restricted adaptation, not an equivalent reimplementation of the entire original function:

- Each row must have exactly one valid image placeholder. Visual features are supplied; image encoding is excluded.
- Text embeddings are supplied, and an explicit `image_mask` identifies the placeholder. Original token IDs and model embedding lookup are unnecessary.
- Padding is always on the right. The original optional length truncation and other padding direction are excluded.
- All five output tensors are always returned. `mm_token_type_ids` is an addition, not a claim that the original function returns it.
- Labels move with their segments; visual labels become `-100`. No causal label shift is performed.

This differs from the modern HF path where a processor expands the placeholder into `N` image markers before the model replaces their embeddings with `masked_scatter`. That scatter already has enough positions available. This task starts with one placeholder and must change the sequence length; a same-length replacement in the original sequence is insufficient.

## Reference approach

First verify that each row has one image marker and reject every marker in an invalid position. Validate counts per row, not just the total across the batch.

For row `b`:

1. Use `attention_mask[b]` to gather valid embeddings, labels, and image markers in order.
2. Find the unique placeholder index `i` in the compressed row, separating the text into a prefix and suffix around it.
3. Concatenate the text prefix, all visual vectors, and the text suffix. Concatenate labels in the same way, using a length-`N_b` segment filled with `-100` for the image.
4. Build matching type IDs, setting only the visual segment to `1`.

After finding the maximum expanded length `S`, allocate the five outputs with their padding defaults. Copy each expanded row into its first `S_b` positions, mark those positions valid, and assign position IDs `0..S_b-1`.

Since `N_b>=1`, every expanded row is nonempty. Even when `N_b=1`, discard the placeholder's original label and assign the visual label and type ID.

## Why it works

Padding removal deletes only invalid positions and preserves the relative order of valid ones. The unique placeholder divides the valid sequence into a unique text prefix and suffix. Concatenating those segments around the visual sequence performs exactly the required replacement, with length `A_b-1+N_b`.

Text labels are neither recomputed nor shifted, and all visual labels ignore the loss. Initialized right-padding positions are never overwritten by valid segments. The same length `S_b` controls writes to embeddings, labels, masks, positions, and types, so all five outputs stay aligned.

A text label of `-100` is independent of image identity. Although padding and visual labels also use `-100`, attention masks and type IDs distinguish them.

## Complexity

Let `S = max_b(A_b-1+N_b)`. Reference time complexity is `O(BL+BSD)`, including mask validation and batch output initialization. Space complexity is `O(BL+BSD)`, including temporary validation masks, intermediate expanded rows, and final outputs. No model computation is needed.

## Tests and pitfalls

There are **14 `check_*` functions**, with `check_example.example = True`. The independent oracle does not reuse the reference's per-row slice concatenation. Instead, it visits original valid positions, emits a text event or several visual events, and independently assembles Python lists before converting them into five tensors.

Coverage includes head/middle/tail placeholders, `N_b=1`, unequal visual lengths, left/right padding and internal holes, text labels of `-100`, padding/placeholder sentinel isolation, noncontiguous offset views for all inputs, tensor/list/backing-storage preservation, both floating dtypes, minimum and maximum bounds, randomized comparisons, and missing, extra, padded, or later-row invalid markers.

Common mistakes include taking a prefix of the valid length instead of gathering by mask, shifting labels by one, inferring images from `label=-100`, keeping original position indices, assigning padding type ID `1`, and truncating expanded sequences to input length `L`.

Code and independent tests passed submission-mode self-testing before the example was executed and the documentation written. Result: **14/14 AC**.

```text
.venv-backend/bin/python scripts/add_problem.py --selftest mm-image-token-expand --mode submit
```
