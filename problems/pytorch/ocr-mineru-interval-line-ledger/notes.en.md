# Solution notes: MinerU interval exclusion, overlap gates, and span-to-line grouping ledger

## Provenance

- Repository `opendatalab/MinerU`, default branch `master` (not `main`). WebSearch hits from deepwiki describe the old fork `chukonu-team/MinerU_2.5` whose paths have changed; the main repository source is the only authority.
- Tree fetched via GitHub API `git/trees/master?recursive=1` (1521 blobs / 635 .py files, truncated=False), then files downloaded with `contents?ref=master` + base64; the `raw.githubusercontent.com` channel proved unreliable in the 833 cycle and is not used.
- Core sources: `mineru/model/ocr/geometry.py` (200 lines) and `mineru/backend/analysis/pdf/text/lines.py` (126 lines), plus threshold conventions in `mineru/backend/analysis/pdf/constants.py` and the ContentType enum in `mineru/types.py`.
- The official cross-check script execs the official functions verbatim via `ast.get_source_segment` over 1700 randomized inputs, all identical. Note that `group_spans_to_lines` hard-codes `threshold=0.6` internally and does not forward it, so the harness must fix 0.6.

## Mechanism highlights

- Overlap gate: `max(0, min(end) - max(start))` clamp, `min(extent)` as denominator, strict `>`, and no division by zero when `min_extent <= 0`. Default 0.8; `lines.py::merge_spans_to_line` explicitly passes 0.6 while `update_det_boxes` uses the default 0.8 — both must be covered separately.
- Interval ledger: `merge_intervals` sorts by start, uses a strict `<` to decide non-overlap (adjacent closed intervals merge), and extends the right end with `max`; `remove_intervals` uses ±1 discrete integer-point semantics, so mask `[2,4]` consumes points 2/3/4, the front segment ends at 1 and the next starts at 5.
- Line grouping: `interline_equation` is downgraded first; only valid spans enter the denominator; both `h/w > 2` and `vertical/valid > 0.8` are strict `>`; vertical uses outer key `bbox[2]` descending + x-axis overlap grouping + inner key `bbox[1]` ascending, horizontal uses outer key `bbox[1]` ascending + y-axis grouping + inner key `bbox[0]` ascending; the special-type break is a bidirectional OR of "itself or any member of the current line".
- Detection box merge: `calculate_is_angle` averages the left and right side heights and treats `not (0.8h <= dy <= 1.2h)` as angle; `points_to_bbox` takes `p1.x/p1.y/p2.x/p3.y`; an aspect ratio strictly `>` 4 triggers `merge_overlapping_spans` for that line (which itself uses a strict `<` on x adjacency); angle boxes are only isolated and appended unchanged at the end.

## Test design

- Four `check_*` functions, each with an independent oracle (pure Python rewrite, no import of solution) + grid cross-check + hand-computed anchors.
- The anchors deliberately pin every strict-vs-non-strict boundary: ratio exactly at threshold, negative-overlap clamp for fully separated y boxes, adjacent closed intervals `[0,10]`/`[10,20]`, ±1 exclusion, `h/w == 2`, vertical ratio exactly 0.8, inner spans whose x0 and y0 orders are opposite, aspect ratio exactly at threshold, and x boundaries exactly touching.
- All three defaults (`threshold=0.6`, `ratio_threshold=4`, `overlap_threshold=0.6`) have separate anchors that omit the parameter; otherwise mutants changing them to 0.8 / 0 would go undetected.

## Mutation design

- 43 single-point mutants; first round 33/43 CAUGHT, 10 SURVIVED: one was a duplicated anchor string (`'"overlap": overlap, "min_extent"'` appears once in each of two returns), nine were genuine coverage gaps.
- Fixes: a fully y-separated bbox pair, span sets with `h/w` exactly 2 and vertical ratio exactly 0.8, a span set with opposite x0/y0 inner order, a poly whose aspect ratio exactly equals the threshold, a double poly with touching x boundaries, and three calls omitting default parameters; the anchor became `'"overlap": overlap,'` with count=2 to replace both sites at once. Final: 43/43 CAUGHT, 0 SURVIVED.
- One hand anchor was written backwards (`h/w == 2` mistakenly treated as a vertical span), which made the correct solution fail too and produced false-positive "all CAUGHT"; after correcting to `vertical_span_count == 0`, both judge modes and mutation passed together. Lesson: a green mutation run cannot replace a self-check of the correct solution.

## De-duplication boundaries

- Different track from `ocr-layout-column-sort` (full-page two-column reading-order state machine), `ocr-deepseek-tile-ledger` (dynamic-resolution tiles and token sequence ledger), `ocr-olmocr-image-merge` (union-find tolerance merge), and `ocr-dbnet-unclip-thresh` (DB postprocess single box): this problem takes only the pure interval and line-grouping ledgers in geometry/lines, with no full-page ordering, no raster, and no pyclipper.
- No overlap with `grid-sample` / `upsample-bilinear` interpolation problems or data-tokenization segmentation problems either.
- Out of scope: real detection forward passes, PIL/torchvision, numpy four-point arrays, LLM calls for title leveling, LaTeX delimiter splitting, cross-page table merging, and markdown rendering.
