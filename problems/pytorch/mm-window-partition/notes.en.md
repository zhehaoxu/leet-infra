# In Detail: Swin Window Partition

## Source
`window_partition` of Liu et al., *Swin Transformer* (arXiv:2111.03861): cut a \((B,H,W,C)\) feature map into non-overlapping \((w,w)\) windows and flatten them into a batch of windows for intra-window self-attention. It is a pure memory rearrangement.

## Core points and common mistakes
1. **Window order is (batch, window row, window col) row-major**: `view(B, H//w, w, W//w, w, C).permute(0,1,3,2,4,5).reshape(-1,w,w,C)` — the permute moves "window row, window col" ahead of "in-window row, in-window col".
2. **The window keeps its (row, col, C) spatial layout**: it is not flattened into 1-D tokens; it remains a \(w\times w\times C\) mini feature map.
3. **Divisibility is required**: the original Swin implementation pads when it does not divide evenly; this problem stipulates that \(H,W\) are multiples of `window` and that **no padding** is done.
4. Difference from patchify: here the last dimension is the channel C and the window keeps its 2-D shape; patchify flattens everything into (C·p·p) features.

## Cross-check convention
The reference solution is `view→permute(0,1,3,2,4,5)→reshape`. The grading `_oracle` uses a four-level Python loop filling elements by window coordinates (different structure, numerically equivalent), with `allclose`. It also asserts "output shape = (B·(H/w)(W/w), w,w,C)", "window merge inverse restoration == the original image", and "the contents of window 0".

## Complexity
\(O(BHWC)\) rearrangement.
