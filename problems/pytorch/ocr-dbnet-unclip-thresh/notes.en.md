# notes — ocr-dbnet-unclip-thresh (order 718)

## Source

- Authoritative source: PaddlePaddle/PaddleOCR `ppocr/postprocess/db_postprocess.py` (main, 2026). DBPostProcess defaults: thresh=0.3, box_thresh=0.7, min_size=3, unclip_ratio=2.0, score_mode="fast".
- **Three-library semantics verified experimentally** (throwaway env with shapely 2.2.0 + pyclipper 1.4.0 + cv2 5.0.0; none exist in the judging env):
  - shapely `Polygon.area` = |shoelace| (vertex-order independent, cw positive too); `Polygon.length` = closed perimeter; degenerate (collinear) area=0.
  - pyclipper (integer Clipper): **AddPath input coordinates truncated toward zero** (C cast; 10.7 -> 10), **Execute output coordinates rounded half-away-from-zero** (floor(v+0.5); 50.5 -> 51, -10.9 -> -11 — opposite of numpy's banker 50.5 -> 50). JT_ROUND on a rectangle yields corner-arc polygons (5-24 points depending on distance), and **the bbox extreme equals trunc(input edge) ± distance then rounded — exact for integer boxes**; non-integer box coordinates carry ±1 arc discretization error (probe: box [10.7,50.9], dist 3.3 -> x_hi 53 vs theoretical 54), so test cases pin integer/half-integer coordinates.
  - cv2 `fillPoly`: rasterizes after astype(int32) truncation toward zero; an axis-aligned integer rectangle covers the **fully closed region** (including right/bottom edges; probe [2,6]×[1,4] -> mask_sum 20 = full 5×4); negative coordinates truncate toward zero (-3.7 -> -3) with out-of-mask parts clipped; single-point polygon sum=1. `cv2.mean(img, mask)` = masked mean, and **an all-zero mask returns 0.0**.
  - `cv2.minAreaRect` is exact for axis-aligned integer input; boxPoints carries float32 noise (-9e-17) and rotated rectangles carry float32 size error — rotated boxes stay out of scope (notes).
- **Officially reachable domain**: with contour coordinates in [0,W-1]×[0,H-1], ceil(max) <= W-1 always holds — the bbox upper clip is dead defense; likewise the lower clip. The problem widens the domain (negative coordinates) so the lower clip takes effect; upper-bound overflow would IndexError (official numpy slices silently truncate) — no such inputs are generated.
- **Gate order**: sside_in gate -> box_score_fast -> box_thresh gate -> unclip (distance/expansion) -> sside_out gate -> dest rescaling. On gate1/box_thresh failure later ledger keys are None (official continue semantics); an all-zero mask (box fully off-image) gives score=0.0, eaten by the box gate.
- Official `if len(box) > 1: continue` (drop multi-path Execute) never fires for normal axis-aligned input; noted, not modeled.
- The dest scaling denominator is the **prob map's W/H** (bitmap.shape), not the source image size; round is numpy banker's rounding.

## Test Design

- Oracle = independent vectorized numpy implementation (meshgrid mask, np.roll shoelace/perimeter, np.rint scaling); score/distance tolerance 1e-9 (ULP differences from summation order in the same float64).
- Constant regions always use binary-exact values (0.75/0.5/0.25) — summing 0.8 sixteen times gives 0.8000000000000002 and `== 0.8` fails (first version got bitten).
- Banker's anchor: expanded coordinates are only bx_lo/bx_hi; construct bx_hi/W*dw = 0.5 (box [-1,0], ratio 4 -> bx_hi=1, W=6, dw=3) -> round=0, while a half-up mutation gives 1.
- sside_out exactly-4 case (ratio 0.225): the +1 span mutation (span mistaken for pixel count) crosses the min_size+2 gate -> caught; the exactly-3 span case is insufficient (4 still < 5).
- 200 randomized cases: coordinates integer or .5 halves (trunc exact); ratio/gates/dest randomized; the perim=0 (min_size=0 + single point) ValueError path anchored separately.

## Mutations (15/15 caught + 1 equivalent)

- Caught: thresh strictness, bbox floor->ceil, bbox lower clip-off (negative index wrap), mask trunc->floor (negative coords), mask union bbox (the two coordinate ledgers), mask dropping the top row, |shoelace| removal (cw), perimeter missing the closing edge, distance halved, pc_round floored, sside_out +1, min_size+2 -> +1, banker->half-up, dest vertical clip-off, box_thresh > -> >=.
- 1 equivalent mutation (SURVIVED, recorded): swapping min/max order in the bbox upper clip — mathematically equivalent.
- **Lesson**: the first _compare zeroed score/distance after isclose and then compared whole dicts — the returned value was corrupted and all downstream hand-computed assertions failed (keep comparison and return separate).

## Pitfalls

- A case box exceeding the prob map (y1=4 with H=4) -> loop IndexError in solve (official cv2 slices silently) — boxes restricted to the official reachable domain.
- Hand-computed assertions using the wrong ratio (1.5 vs the default 2.0) caused two rounds of false failures — verify anchor assertions against case parameters in the same breath.
- Boundary with 668 (ctc-greedy) / 691 (layout-column-sort) / 692 (olmocr-merge): this problem owns the DB detection postprocess score/expansion/gates; cv2/shapely/pyclipper are absent from the judging env, hence full scalarization.
