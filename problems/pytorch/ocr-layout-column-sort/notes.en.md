# notes — ocr-layout-column-sort (order 691)

## Source

- Official code: PaddlePaddle/PaddleOCR `ppstructure/recovery/recovery_to_doc.py`,
  function `sorted_layout_boxes(res, w)` (GitHub `main` branch, fetched
  2026-11).
- Context: PP-Structure layout recovery calls it to reorder `res` (layout
  detection results, each with `bbox`/`type`/`res`) into reading order before
  docx export; the docx side then switches Word section column format per
  `region["layout"]` ("single"/"double") via the `flag` state machine in
  `convert_info_docx`. This problem tests only the sorting function.
- Exploration log: the todos originally said "Surya-family or PP-Structure
  XY-cut". Both old and new surya reading order are model-decoded (v0.4.15
  is an encoder-decoder AR emitting ranks step by step; master uses box
  tokens cross-attending with a constrained greedy AR), so no rule ledger
  exists there; only `canonical_order` (raster key `int(y0 // (1000/24))`
  with `(x0, y1, x2)` tiebreak) is usable but too thin — surya was dropped
  in favor of this rule-based PaddleOCR implementation.

## Ledger rules

1. `num_boxes == 1`: return the original list with `layout="single"` set,
   **no sorting**.
2. Sort key `(bbox[1], bbox[0])` = `(y0, x0)`; Python `sorted` is stable, so
   equal keys keep input order (H2 anchor).
3. Left-bucket condition `x0 < w/4 and x1 < 3w/4`; right-bucket condition
   `x0 > w/4 and x1 > w/2`; all strict:
   - `x0 == w/4`: neither condition holds (not `<` and not `>`) → single
     (anchor D).
   - `x1 == 3w/4`: the left condition fails; with `x0 > w/4` also false the
     box becomes single (anchor E).
   - Coverage gap: a full-width box with `x0 < w/4` and `x1 > 3w/4` fails
     both conditions → single, which is exactly the intended "full-width
     block as separator" behavior.
4. Last box: three branches plus a drop:
   - The spanning condition `y0 > prev.y1 and x0 < w/2 and x1 > w/2` — note
     `prev` is the previous box **in scan order** (`_boxes[i-1]`), not in
     input order; `>` is strict, so a box with `y0 == prev.y1` does not
     enter the spanning branch (last-gap anchor).
   - The spanning branch flushes **before** appending; the column branches
     append **before** flushing — so the last box ends up outside vs inside
     its column group (anchors I / J).
   - The drop condition `x1 <= w/2 and x0 >= w/2`, combined with `x0 <= x1`,
     forces `x0 == x1 == w/2` (degenerate zero-width box). **No flush
     happens on drop** — accumulated buckets vanish with it (anchor F
     returns `[]`). This is the most hidden official behavior.
5. Loop exit analysis: `i` grows by 1 each iteration, so for
   `num_boxes >= 2` the loop always reaches the last-box branch and breaks,
   with buckets flushed or cleared inside; `num_boxes == 0` exits via
   `i >= num_boxes` with empty buckets. The trailing `if res_left /
   if res_right` flush after the loop is therefore **unreachable dead code**
   — removing it is an equivalent mutation and was not collected.

## Oracle design

The `_oracle` in test_cases is an independent rewrite: hand-written
insertion-based stable sort (no `sorted`), explicit `queue.pop(0)` scanning,
`pending_left/right` buckets, and an `(idx, layout)` tuple stream. Compared
against `solve` on 300 random cases (w in {400,800,1000,1200}, integer
coords). Random coordinates almost never produce `y0 == prev.y1` with a
spanning last box (the lesson from `last-gap-inclusive` surviving round
one), so the targeted anchor
`check_last_gap_touching_not_spanning_branch` pins it.

## Mutation record

All 13 caught: sort-key-swapped, left-cond-half, left-cond-inclusive,
right-cond-3q, cols-swapped, last-gap-inclusive (caught by the added
touching-boxes anchor), span-cond-moved, flush-after-append,
last-right-then-left, drop-becomes-single, single-not-flushed,
single-box-sorted-out, left-right-flip-last. No surviving equivalent
mutations (the dead trailing flush was not mutated).

## Dedup

- Different from `ocr-layoutlm-bbox-index` (bbox normalization / 2D position
  indexing): that problem maps coordinates to indices; this one is a
  sorting state machine.
- Boundary with 347 task DAG: no graph here, rules are purely geometric.
- `w/4`, `3w/4`, `w/2` are float divisions; all test widths are multiples of
  4 so anchors carry no float ambiguity.
