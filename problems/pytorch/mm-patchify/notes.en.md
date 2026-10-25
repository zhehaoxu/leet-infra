# In Detail: ViT Image Patchify

## Source
Dosovitskiy et al., *An Image is Worth 16×16 Words* (arXiv:2010.11929): cut the image into non-overlapping patches and flatten them into a token sequence. This problem only performs the geometric rearrangement (the linear projection is a subsequent `nn.Linear` and is not involved).

## Core points and common mistakes
1. **Token order = block-row major**: iterate block rows first, then block columns (`permute(0,2,4,...)` brings block_row, block_col to the front), not channel-major and not column-major.
2. **In-token flattening order (C, row, column)**: the channel is the outermost dimension. Different conventions would misalign the subsequent projection weights — this problem fixes it to (c,i,j).
3. **The reshape splits must map H→(nr,p), W→(nc,p)**: `reshape(B,C,nr,p,nc,p)` relies on contiguous memory; call `.contiguous()` first for non-contiguous inputs.
4. It is a **bijection**: mutually inverse with unpatchify, so no element is lost or duplicated.

## Cross-check convention
The reference solution is `reshape→permute(0,2,4,1,3,5)→reshape`. The grading `_oracle` uses a five-level Python loop to fill elements according to the convention (completely different structure, numerically equivalent), with `allclose(atol=1e-5,rtol=1e-5)`. It also asserts "shape is correct", "unpatchify inverse restoration == the original image", "the number of blocks N=(H/p)(W/p)", and "the values of a single block".

## Complexity
\(O(BCHW)\) rearrangement.
