# CornerNet Corner Pooling Forward Ledger

## Background

CornerNet (1808.01244) drops RoIs and models top-left/bottom-right object corners as heatmap peaks. Corner pooling lets a corner "see" inside the box: for a top-left corner, take per-point max to the **right** and max **below**, then add. The official implementation lives in `_cpools` (C++/CUDA four-direction pools); this problem transcribes the forward pass as a pure ledger.

## Exact Ledger

1. **Four-direction running max** (per row/column, scan includes self):
   - `right_pool` (official right_pool.cpp): scan rightward accumulating, `out[i][j] = max(x[i][0..j])` — takes the **left-side** max. The filename names the scan direction; semantically it takes the left max — counter-intuitive.
   - `left_pool`: scan leftward, `out[i][j] = max(x[i][j..W−1])` — right-side max.
   - `top_pool`: scan downward, `out[i][j] = max(x[i..H−1][j])` — bottom-side max.
   - `bottom_pool`: scan upward, `out[i][j] = max(x[0..i][j])` — top-side max.
2. **Corner combinations** (official models/CornerNet.py: `tl_pool = (TopPool, LeftPool)`, `br_pool = (BottomPool, RightPool)`):
   - tl = top + left (bottom max + right max — matches the paper's "max to the right + max to the bottom")
   - br = bottom + right; tr = top + right, bl = bottom + left (official implements only tl/br; tr/bl by symmetry)
3. **Each branch pools separately, then elementwise add** — not add-then-pool.
4. Structural: out ≥ x elementwise, same shape, O(HW) scan.

## Constraints

1. x is a non-empty rectangular [H][W] of finite floats; mode ∈ {tl,tr,bl,br}.
2. Single row/column degenerates legally (top/bottom branch = self on a single row).
3. No input mutation; float list output.

## Data Scale

- H, W ≤ 512 (O(HW) ledger).

## Acceptance Anchors (excerpt)

- `x = [[1,5,2],[3,0,4]]`: tl = `[[8,10,6],[7,4,8]]` (top=`[[3,5,4],[3,0,4]]` + left=`[[5,5,2],[4,4,4]]`); br = `[[2,10,7],[6,8,8]]`.
- All-2.0 input → all-4.0 output for every mode.
- All-negative input: running max still includes self (bl anchor in tests).
