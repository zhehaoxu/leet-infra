# PaddleOCR DB Postprocess Single-Box Ledger: box_score_fast and unclip

## Problem

DBNet's detection head outputs a probability map; post-processing turns binarized regions back into text boxes: threshold binarization -> contours -> minAreaRect -> region score gate -> unclip expansion (Shapely/Clipper) -> size gate -> coordinate rescaling. This problem hand-writes the full single-box quad ledger of PaddleOCR's `DBPostProcess.boxes_from_bitmap`: the contour is already reduced to an axis-aligned rectangle (4 points) by `get_mini_boxes`; no real images and no calls to cv2/shapely/pyclipper — all three libraries' semantics were pinned by experiment and reproduced in pure scalars.

## Solution

Official behavior (PaddlePaddle/PaddleOCR `ppocr/postprocess/db_postprocess.py`, pinned line by line; shapely 2.2 + pyclipper 1.4 + cv2 5.0 verified):

- **Binarization**: `segmentation = pred > thresh` (strict, pointwise);
- **sside gate first**: `get_mini_boxes`' `sside = min(span_x, span_y)`; `sside < min_size` (default 3) -> skip, score/distance/expanded all uncomputed;
- **box_score_fast**: bbox per axis `xmin=clip(floor(min x),0,W-1)`, `xmax=clip(ceil(max x),0,W-1)` (y likewise with H); the mask comes from `cv2.fillPoly` rasterization — coordinates are **truncated toward zero** by `astype(np.int32)` (-3.7 -> -3, not floor's -4) and form a closed polygon (axis-aligned rectangle = closed rectangle including right/bottom edges), intersected with the bbox region; `score = sum(prob·mask)/mask_sum`, and **an all-zero mask gives cv2.mean = 0.0** (box fully off-image);
- **Score gate**: `box_thresh > score` -> skip (strict; a tie passes);
- **Distance**: `|shoelace| × unclip_ratio / closed perimeter` (shapely `Polygon.area` is vertex-order independent via absolute value; `Polygon.length` includes the closing edge);
- **pyclipper JT_ROUND expansion**: **input coordinates truncated toward zero** (10.7 -> 10), **output coordinates rounded half-away-from-zero** (`floor(v+0.5)`; 50.5 -> 51, opposite of numpy's banker 50); axis-aligned expansion = `x_lo = r(trunc(min x) − distance)`, `x_hi = r(trunc(max x) + distance)` (y likewise);
- **Size gate**: after unclip, `sside < min_size + 2` -> skip;
- **Dest rescaling**: `clip(round(x/W*dest_width), 0, dest_width)` — numpy round is **banker's rounding** (6.5 -> 6, 19.5 -> 20), with W the prob map width.

## Data Scale

- 10 hand-computed anchors: basic (dist=1.5 exact, expanded [0,6]²), cw vertex order (|shoelace|), pred equal to thresh not counted, floating-point rectangle (bbox floor/ceil and mask truncation coexist: bbox=[2,7,1,6] vs mask [2,6]×[1,5]), negative coordinates (bbox clipped to 0 plus fillPoly truncation, 2×3 px intersection), edge-touching, fully off-image box (score=0.0 -> box gate), single-point degenerate (sside=0 -> min_size_in), slender-box sside_out gate (spans 3 and exactly 4), score exactly equal to box_thresh passing, banker's 0.5->0 anchor (1/6*3=0.5) and two-sided clip;
- 200 randomized comparisons (oracle = independent vectorized numpy implementation; coordinates restricted to the officially reachable domain: integers or .5 halves).

## Scoring

All twelve ledger keys checked: pos_count/sside_in/bbox/score/distance/expanded/sside_out/gate/passed/scaled exactly, score/distance at 1e-9; the bitmap matrix pointwise.

## Red Lines

- No import of cv2 / shapely / pyclipper (AST-level check); pure scalar ledger;
- Coordinates restricted to the officially reachable semantics (integers or floats unchanged by truncation — non-integral floats hit JT_ROUND arc discretization error of ±1, kept out of scope; see notes).
