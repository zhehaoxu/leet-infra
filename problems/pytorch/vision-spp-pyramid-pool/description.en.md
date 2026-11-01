# SPP-Net Pyramid Pooling Ledger

## Problem

SPP-Net pools arbitrary H×W feature maps into a fixed length via a cascaded
[1,2,4,...] spatial pyramid. The official caffe implementation does not
compute bin boundaries directly; it translates each level into a max-pooling
kernel/stride/pad — producing windowing that differs from the paper
pseudocode: windows may overlap, and empty bins can occur. Given a feature
map and per-level bin counts, reproduce the official ledger.

Given:
- `feature`: an H×W 2-D float list
- `bin_counts`: per-side bin counts per level (e.g. [1,2,4], processed in
  the given order)

## Per-level ledger (bin count k)

1. `kernel = ceil(H/k)` (H and W computed separately, may differ);
   `stride = kernel`; `pad = (kernel·k − H + 1) // 2` (symmetric remainder).
2. Window of bin `(ph, pw)`:
   `hs = max(0, ph·kernel − pad)` (start may be negative, clamp to 0);
   `he = min(hs + kernel, H)` (end uses the **unclamped** hs+kernel, then
   clamps to H). W axis analogous (kernel_w/pad_w).
3. An empty window (hs≥he or ws≥we) outputs −inf (official −FLT_MAX).
4. Window max, flattened in (ph,pw) row-major order and concatenated level
   by level.

## Output

`{"pooled": [...], "shapes": [[k,k]...], "kernels": [[kh,kw]...],
"pads": [[padh,padw]...]}`.

## Input constraints

- feature is a regular non-empty finite 2-D list; bin_counts non-empty with
  ints ≥1; H, W ≥ max(bin_counts); violations raise `ValueError`.

## Example

5×5 + [1,2,4] → 21 dims: k=1 level kernel=5, pad=0, global max=24; k=2
level kernel=3, pad=1, bin(0,0) window (0..3)×(0..3) → 12; k=4 level
kernel=2, pad=2, bin(0,0) window (0..2)×(0..2) → 6.
