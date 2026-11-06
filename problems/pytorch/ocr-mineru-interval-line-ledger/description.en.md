# MinerU: interval exclusion, overlap gates, and span-to-line grouping ledger

## Background

OCR post-processing in `opendatalab/MinerU` (`master`) contains four pure numeric/integer stages independent of the detection model itself: the y/x overlap-ratio gate on bboxes, closed-interval sorting/merging with mask exclusion, span-to-line grouping (including vertical-text detection and special-type line breaks), and angle-box isolation with aspect-ratio-triggered horizontal merging of detected four-point boxes. This problem distills those four stages into **stateless judging**: a bbox is always an int list `[x0, y0, x1, y1]`, a `poly` is always 4 `[x, y]` points, and span types are always strings; no real detector, no image loading, no torch/PIL. Implement a single entry point `solve(mode, **kw)` with 4 modes.

The official `master` branch has been refactored; the old `span_block_fix.py` / `para_split.py` paths are gone. This problem only takes the deterministically reproducible pure ledgers from `mineru/model/ocr/geometry.py` and `mineru/backend/analysis/pdf/text/lines.py`.

## General conventions

Every mode returns a dict; field semantics are in "Modes and rules", and unlisted intermediate values are not judged. Thresholds / constants are all explicit inputs and judging passes them explicitly; the defaults below are only for reference:

```
OVERLAP_DEFAULT = 0.8
LINE_OVERLAP = 0.6
VERTICAL_RATIO = 2
VERTICAL_DOMINANCE = 0.8
LINE_WIDTH_TO_HEIGHT = 4
SPECIAL_TYPES = {"interline_equation", "image", "table"}
```

Two global rules: **(1) every ratio gate is a strict `>` — equality with the threshold never passes; (2) `lines.py::merge_spans_to_line` explicitly passes `threshold=0.6` to the overlap gate, while `update_det_boxes` uses the default `0.8`; the two thresholds differ and both must be covered.**

## Modes and rules

### 1. overlap_gate: y/x overlap-ratio gate

Inputs: `bbox1, bbox2, axis="y", threshold=0.8`. Returns `{"overlap", "min_extent", "ratio", "exceeds", "zero_extent"}`.

```
y: overlap = max(0, min(y1_1, y1_2) - max(y0_1, y0_2));  e = min(h1, h2)
x: overlap = max(0, min(x1_1, x1_2) - max(x0_1, x0_2));  e = min(w1, w2)
if e <= 0: ratio = None; exceeds = False; zero_extent = True
else:      ratio = overlap / e; exceeds = ratio > threshold; zero_extent = False
```

- The `max(0, ...)` clamp is mandatory: when boxes are fully separated the raw difference is negative and must return to 0, with `ratio` equal to `0.0` (not a negative number).
- Degenerate boxes (zero width or height) never divide by zero; they return `zero_extent=True` directly.
- When `ratio` exactly equals `threshold`, `exceeds=False` (strict `>`).

### 2. interval_ledger: closed-interval merge and ±1 mask exclusion

Inputs: `intervals, masks, original` (`original` may be `None`). Returns `{"merged_intervals", "merged_masks", "remaining", "n_segments", "covered_points", "remaining_points"}`.

```
merge_intervals(ivs): sort by start; if not merged or merged[-1][1] < it[0]: append
                      else: merged[-1][1] = max(merged[-1][1], it[1])
remove_intervals([s,e], masks):
    for (ms, me) in merge_intervals(masks):
        if ms > e or me < s: continue
        if cur < ms: remaining.append([cur, ms - 1])
        cur = max(me + 1, cur)
    if cur <= e: remaining.append([cur, e])
```

- The merge condition is a **strict `<`**: adjacent closed intervals `[0,10]` and `[10,20]` are merged into `[0,20]`.
- Exclusion uses **±1 discrete integer-point** semantics: mask `[2,4]` consumes points 2/3/4, so the front segment ends at `1` and the next starts at `5`.
- With `original=None`, `remaining=[]`, `covered_points=0`, `remaining_points=0`; otherwise `covered_points = (e-s+1) - sum(points per segment)`.
- `merged_intervals` and `merged_masks` are computed on **input copies**; in-place side effects are not judged.

### 3. line_grouping: span-to-line grouping

Inputs: `spans` (each `{"bbox", "type"}`), `threshold=0.6`. Returns `{"vertical", "valid_span_count", "vertical_span_count", "downgraded", "lines", "line_bboxes", "orientation"}`.

```
1) interline_equation -> inline_equation, count downgraded
2) skip bbox if len < 4 or w <= 0 or h <= 0; valid += 1; h/w > 2 => vertical += 1
3) is_vert = valid > 0 and vertical / valid > 0.8
4) vertical:   outer key bbox[2] desc, group by x-axis overlap, inner key bbox[1] asc
   horizontal: outer key bbox[1] asc,  group by y-axis overlap, inner key bbox[0] asc
5) force break if self.type in SPECIAL_TYPES or any member of current line is
6) line_bbox = [min x0, min y0, max x1, max y1] over the line
```

- Both `h/w > 2` and `vertical/valid > 0.8` are **strict `>`**: `h/w == 2` is not a vertical span, and an exact 0.8 ratio is not judged vertical.
- The denominator of the vertical test is the **valid span count** (degenerate boxes are excluded from `valid`), not the total span count.
- The special-type break is a bidirectional OR of "itself or any member of the current line"; `interline_equation` is already downgraded, so it no longer triggers a break.
- Inner sort key: `bbox[0]` (x0) for horizontal, `bbox[1]` (y0) for vertical — different from the outer grouping key.
- Default `threshold=0.6` (not 0.8): `lines.py::merge_spans_to_line` hard-codes 0.6 and does not forward an external threshold.

### 4. det_box_merge: angle isolation and aspect-ratio-triggered merge

Inputs: `polys`, `ratio_threshold=4`, `overlap_threshold=0.6`. Returns `{"angle_flags", "horizontal_bboxes", "line_bboxes", "ratio_triggered", "result_bboxes"}`.

```
is_angle(poly):  height = ((p4.y - p1.y) + (p3.y - p2.y)) / 2
                 return not (0.8*height <= (p3.y - p1.y) <= 1.2*height)
points_to_bbox:  [p1.x, p1.y, p2.x, p3.y]
per poly: angle=True -> angle_polys, else -> points_to_bbox into horizontal
horizontal sorted by y0, grouped by y-axis overlap (overlap_threshold)
per line: lb = bounding box of the line
          trig = (lb.w) > (lb.h) * ratio_threshold
          trig -> merge_overlapping_spans(line) (strict < on x adjacency)
          else -> keep each bbox unchanged
append points_to_bbox(p) for every p in angle_polys at the end
```

- `is_angle` uses the **average of the left and right side heights**; the test is `not (0.8h <= dy <= 1.2h)`, i.e. a dy inside the band is non-angle.
- `points_to_bbox` takes `p1.x / p1.y / p2.x / p3.y` (not any other point combination).
- The aspect-ratio test is a **strict `>`**: `w == 4*h` does not trigger.
- `merge_overlapping_spans` uses `merged[-1][2] < b[0]` (strict `<`), so two boxes whose x boundaries exactly touch are merged.
- Angle boxes **do not** take part in line grouping or the aspect-ratio test; they are appended unchanged at the end.
- Defaults: `ratio_threshold=4`, `overlap_threshold=0.6`.

## Input and output examples

```
solve(mode="overlap_gate", bbox1=[0, 0, 10, 10], bbox2=[0, 5, 10, 15],
      axis="y", threshold=0.8)
# {"overlap": 5, "min_extent": 10, "ratio": 0.5,
#  "exceeds": False, "zero_extent": False}

solve(mode="interval_ledger", intervals=[[5, 8], [0, 3], [7, 12]],
      masks=[[2, 4], [6, 9]], original=[0, 12])
# {"merged_intervals": [[0, 3], [5, 12]],
#  "merged_masks": [[2, 4], [6, 9]],
#  "remaining": [[0, 1], [5, 5], [10, 12]], "n_segments": 3,
#  "covered_points": 7, "remaining_points": 6}

solve(mode="det_box_merge",
      polys=[[[0, 0], [100, 0], [100, 10], [0, 10]],
             [[5, 1], [105, 1], [105, 9], [5, 9]]],
      ratio_threshold=4, overlap_threshold=0.6)
# {"angle_flags": [False, False],
#  "horizontal_bboxes": [[0, 0, 100, 10], [5, 1, 105, 9]],
#  "line_bboxes": [[0, 0, 105, 10]], "ratio_triggered": [True],
#  "result_bboxes": [[0, 0, 105, 10]]}
```

## Out of scope

Real layout/VLM/OCR detection forward passes, PIL/torchvision, numpy four-point arrays and `sorted_boxes` full-page reading order (already covered by `ocr-layout-column-sort`, `ocr-deepseek-tile-ledger`, etc.), the LLM call for title leveling, `_natural_language_spans` LaTeX delimiter splitting, cross-page table merging and markdown rendering. Judging covers only the four pure numeric/integer ledgers above.
