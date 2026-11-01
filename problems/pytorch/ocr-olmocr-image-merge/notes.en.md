# notes — ocr-olmocr-image-merge (order 692)

## Source

- Official code: allenai/olmocr `olmocr/prompts/anchor.py`, function
  `_merge_image_elements(images, tolerance=0.5)` (GitHub `main`, fetched
  2026-11; the accompanying `tests/test_anchor.py` only covers the final
  anchor text, not the merge details).
- Call chain: `get_anchor_text(pdf, page, "pdfreport")` → `_pdf_report`
  (pypdf `extract_text` visitors collect `TextElement`/`ImageElement`;
  image bboxes come from CTM-transforming the unit square `[0,0]-[1,1]`) →
  `_linearize_pdf_report`, which merges images first and then linearizes
  under a length budget.
- Scoping decisions: `_linearize_pdf_report`'s over-budget path contains an
  unseeded `random.shuffle` (not reproducible), `_cleanup_element_text`
  depends on `ftfy` (unavailable in the judge environment), and
  `_cap_split_string`'s head/tail truncation quirks are fun but thin — none
  are tested. `_merge_image_elements` is a fully deterministic union-find
  ledger, so it became the problem.

## Ledger rules

1. Separation distance: `h_dist = max(0, max(x0s) - min(x1s))`. Separated
   boxes give `max(x0s) >= min(x1s)` → positive gap; overlapping/touching
   boxes give negative or zero, clamped to `<= tol`. **Closed interval**
   `<= tolerance` (gap exactly equal to tolerance merges; anchors 2b/2c
   confirm 0.5 merges, 0.6 does not).
2. Equivalence of the `max(0, ...)` clamp: a negative distance (real
   overlap) stays negative without the clamp and still satisfies
   `<= tol` → removing the clamp is a **mathematically equivalent
   mutation** and was not collected (explained here). Swapping min/max
   (`min(x0s) - max(x1s)`) flips positive gaps negative → immediately
   observable.
3. Union direction `parent[root_i] = root_j` (i's root attaches to j's
   root) → the final component root skews toward later elements, but
   `groups` inserts by `find(i)` over 0..n-1 → **output order = ascending
   min index per component** (independent of which root wins). Hence
   reversing the union direction or toggling path compression are
   equivalent mutations, not collected.
4. Name concatenation: `indices` ascend (appended in scan order via
   `groups.setdefault(...).append(i)`), and
   `merged_name += f"+{name}"` → **concatenation in ascending index
   order**. Anchor 4 pins `["z"(0.8,...), "a"(0,0,1,1)]` → `"z+a"`: the
   union outcome cannot change it.
5. The bbox aggregation starts from `images[indices[0]]` and expands per
   member — equivalent to the per-axis min/max over all members.
6. The pairwise scan is all pairs `i<j` (O(n²) adjacency test), not just
   adjacent pairs — the `adjacent-pairs-only` mutant is caught by
   skip-overlap configurations (random checks plus the chain anchor).

## Oracle design

`_oracle` uses an adjacency list + explicit-stack BFS for connected
components (no union-find); components discovered in ascending start order
are naturally sorted by min index, and aggregation uses generator
min/max. Compared against `solve` on 300 random cases (1-decimal
coordinates, tolerance in {0, 0.25, 0.5, 1.0}), containing many
touching/exactly-at-tolerance boundary configurations.

## Mutation record

All 13 caught: tol-exclusive, dist-and-to-or, h-dist-minmax-swap,
v-dist-ignored, adjacent-pairs-only, name-no-plus, name-last-wins,
name-reversed-indices, bbox-first-only, bbox-minmax-x-swap,
tolerance-doubled (always-true conjunction), self-union-only (round one's
replacement hit the `def union(i, j)` signature and raised SyntaxError;
re-scoped to the call site and rerun), groups-key-original-index. Two
equivalent mutations (removing the max(0,...) clamp; reversing union
direction) were analyzed and not collected, reasons above.

## Dedup

- No union-find/connectivity problem exists in the repo (`grep union|dsu|merge`
  hits are all weight merges / patch merges).
- Adjacent same-day problem `ocr-layout-column-sort` (order 691): that one
  is a rule-based sorting state machine, this one is graph-connectivity
  aggregation; no mechanism overlap.
- `mm-patch-merger`/`mm-prumerge-*` are visual-token merges driven by
  attention scores, different from geometric bbox-tolerance merging.
