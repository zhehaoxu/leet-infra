# Multimodal: Qwen2-VL 2×2 Visual Token Merging

## Problem

The **PatchMerger** of Qwen2-VL (Bai et al., arXiv:2409.12191) uses "2×2 spatial merging" to compress the number of visual tokens to 1/4 while expanding the channel count by 4×: the embeddings of every adjacent \(2\times2\) block in the \(h\times w\) patch grid are concatenated along the feature dimension, after which they go through an MLP projection (this problem only performs this deterministic **spatial shuffle + concat** step, without the weighted Linear).

## Rules

Implement `solve(image_embeds, grid_h, grid_w)`. The input `image_embeds` (float32, shape \((g_h\cdot g_w,\ D)\), the patch sequence flattened in row-major order; \(g_h,g_w\) are even). Return \((\frac{g_h}{2}\cdot\frac{g_w}{2},\ 4D)\): the \((i,j)\)-th output token = the \(D\)-dimensional embeddings of the four patches at grid positions \((2i,2j),(2i,2j\!+\!1),(2i\!+\!1,2j),(2i\!+\!1,2j\!+\!1)\), concatenated in that order. Output tokens are ordered row-major by (merged row, merged column). `einops` is forbidden.

## Examples

\(g_h=g_w=2,\ D=2\), `image_embeds=[[0,1],[2,3],[4,5],[6,7]]` (grid positions (0,0),(0,1),(1,0),(1,1) in order):

```
the only merged block = the four patches concatenated in order
out = [[0,1, 2,3, 4,5, 6,7]]   shape = (1, 8)
```

## Constraints

- CPU float32; both `grid_h` and `grid_w` are even.
- The merge order is fixed as top-left→top-right→bottom-left→bottom-right (i.e. the 2×2 in row-major order).
- Pure rearrangement + concatenation: the output elements are a permutation of the input, with no information loss.
