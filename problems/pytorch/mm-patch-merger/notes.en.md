# In Detail: Qwen2-VL 2×2 Visual Token Merging

## Source
`PatchMerger` of Bai et al., *Qwen2-VL* (arXiv:2409.12191): concatenate the embeddings of adjacent \(2\times2\) visual patches (token count ↓4, channels ↑4), then project them to the LLM dimension through LayerNorm+MLP. This problem only performs the deterministic **spatial shuffle + concat**, stripping away the weighted Linear.

## Core points and common mistakes
1. **The block view must stay aligned with row-major order**: flattening `(gh//2, 2, gw//2, 2, D)` is exactly the grid row-major layout (`(2i+r)*gw+(2j+c)`), and `permute(0,2,1,3,4)` gives (merged row, merged col, r, c, D). Getting it backwards splices the top/bottom or left/right blocks incorrectly.
2. **In-block order is top-left→top-right→bottom-left→bottom-right**: i.e. (r,c)=(0,0),(0,1),(1,0),(1,1), which is the segment order of the concatenated 4D feature.
3. **What is merged is the spatial neighborhood, not four consecutive sequence elements**: two patches on different rows are not in the same 2×2 block.
4. The channel dimension D is innermost, so the output width is 4D.

## Cross-check convention
The reference solution is `view→permute(0,2,1,3,4)→reshape(-1,4D)`. The grading `_oracle` uses a double loop that explicitly gathers the four patches by 2×2 block coordinates and extends them in order (different structure, numerically equivalent), with `allclose`. It also asserts "output shape=(gh/2·gw/2, 4D)", "the output is an element permutation of the input (multisets equal)", and "the value order of a single block".

## Complexity
\(O(g_h g_w D)\) rearrangement.
