# Multimodal: ViT Image Patchify

## Problem

The Vision Transformer (Dosovitskiy et al., *An Image is Worth 16×16 Words*, arXiv:2010.11929) cuts an \(H\times W\) image into non-overlapping \(\text{patch}\times\text{patch}\) tiles, flattens them into a sequence of "visual tokens", and feeds them into the Transformer. This problem implements this **patchify** step (without the linear projection; only the geometric rearrangement).

## Rules

Implement `solve(image, patch)`. Input `image` (float32, shape \((B,C,H,W)\), where \(H,W\) are divisible by `patch`). Return shape \((B, N, C\cdot \text{patch}^2)\), where \(N=(H/\text{patch})\cdot(W/\text{patch})\).

Layout conventions:
- Token order is **block-row major** (first `block_row`, then `block_col`);
- Within each token, the \(C\cdot\text{patch}^2\) dimensions are flattened as **(channel c, in-block row i, in-block column j)** (channel outermost).

Calling `torch.nn.functional.unfold`/im2col directly is forbidden (you may use `reshape`+`permute`).

## Examples

`image` (\(1\times1\times4\times4\)) is the \(4\times4\) matrix filled with \(0..15\), `patch=2`:

```
Input grid          Output 4 tokens (each 4-dimensional)
[[ 0  1 | 2  3]     [0,1,4,5] [2,3,6,7]
 [ 4  5 | 6  7]     [8,9,12,13] [10,11,14,15]
 ----------------
 [ 8  9 |12 13]
 [12 13|14 15]]
out shape = (1, 4, 4)
```

## Constraints

- CPU float32; \(H,W\) are integer multiples of `patch`.
- It is a pure rearrangement (a bijection) with no information loss — the tests compare against the original via the inverse unpatchify operation.
