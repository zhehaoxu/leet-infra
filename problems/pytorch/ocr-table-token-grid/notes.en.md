# notes — ocr-table-token-grid (order 689)

## Upstream

- Source: `PaddlePaddle/PaddleOCR` `ppstructure/table/tablepyxl/tablepyxl.py`, `write_rows` (the official Excel export path of PP-Structure's `TableSystem`; the upstream tablepyxl project is vendored). Attribute parsing `string_to_int`: `s.isdigit() → int(s)`, else **0**.
- Three official mechanisms: ① `cell = ws.cell(row, col)` followed by `while isinstance(cell, MergedCell): col += 1` occupancy skipping; ② on `colspan/rowspan > 1`, `merge_cells(start_row, start_column, end_row, end_column)` with the top-left as anchor; ③ `col += colspan`, then at row end `col = initial_column`, `row += 1`.
- **Three pinned quirks**: non-numeric attribute → 0 → no merge and `col += 0`, so the next cell **overwrites the same slot** (the cells ledger shows two cell_ids on one (row, col)); fullwidth digit `"２"` satisfies isdigit → `int("２") = 2` (Python semantics, real official behavior); non-anchor slots inside a merge are value-cleared in openpyxl (grid stores None).
- **Merge extends rows**: openpyxl `merge_cells` auto-expands the sheet — a rowspan past the last row grows `num_rows` by the merge bottom (the random cross-check caught this bug in the first draft, then the rule and anchor were pinned).
- Boundary vs vision/OCR items: how SLANet produces the structure tokens (model forward) is out of scope; this problem only covers the **token sequence → grid expansion ledger**. TEDS tree edit distance is a different mechanism (evaluation side), not included.
- No openpyxl/lxml in the judging env — input is structured directly as `[text, colspan_attr, rowspan_attr]` (attribute strings as-is); merge/occupancy semantics are replicated in pure Python per openpyxl's storage semantics (pinned in the spec, with the simplification noted: ill-formed tables with overlapping merge regions are out of scope).

## Judging

- solve inputs: rows ([[text, cs_attr, rs_attr], ...], attributes str|None). Returns cells/merges/num_rows/num_cols/grid.
- Oracle: `test_cases._py` uses explicit while-loop pointers + a dict occupancy table + an independent grid backfill (writes first, then clears non-anchors), independent of solve's set-based path; 200 random cross-checks (attribute pool: None/"1"/"2"/"3"/"0"/"abc"/"２"/"").
- Anchors: header colspan=2 → merge [1,1,1,2], h2 lands at (1,3), grid [0,None,1]; first-column rowspan=2 → row-2 b starts at (2,2) skipping occupancy; 2×2 rectangular merge with two rows skipping; `"abc"`/`"0"` attributes → same-slot overwrite; fullwidth "２" → colspan 2; last-row rowspan=3 → num_rows extends to 3; empty table/row; a later row widens num_cols.
- Invalid inputs: rows not a list / td not a 3-tuple / text not str / attribute neither str nor None → ValueError.

## Mutations (11 observable all caught + 2 equivalent)

occupied-skip-off, col-advance-constant, row-no-reset, non-numeric-to-one, merge-end-plus-one, spans-swapped, merge-condition-and, grid-anchor-also-none, cells-zero-based, col-advance-rowspan, none-attr-default-zero — all caught. **Equivalent**: anchor-occupied (the column pointer advances monotonically and row resets change the row key, so a merge anchor is never revisited; adding it to occupied is unobservable); isdigit-to-isdecimal (their differing characters are No-class like `²`, where `int()` crashes — no exception-output contract in this problem).

## Red lines

No real images/model forward/PIL — token sequences are structured inputs; ill-formed tables with overlapping merges excluded (openpyxl overlap semantics vary by version, not pinned); TEDS/table-eval metrics excluded (evaluation-side candidate); `ocr-layoutlm-bbox-index` already owns document bbox position indexing.
