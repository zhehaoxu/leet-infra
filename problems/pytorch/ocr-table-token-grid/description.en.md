# Table Structure Tokens → Cell Grid Allocation Ledger

## Problem

Table recognition models (e.g. PP-Structure SLANet) output **structure token sequences**: HTML made of `<tr>`/`<td>`/`colspan`/`rowspan`. Turning that into a usable spreadsheet requires allocating every cell to grid coordinates — exactly what PP-Structure's official Excel export path, `tablepyxl.write_rows`, does. This problem transcribes that ledger: given per-row cell lists (text plus colspan/rowspan attribute strings as-is), output each cell's (row, col) allocation, the merge regions, and the final ownership grid.

## Solution

The ledger is pinned by the spec (faithful to official `write_rows`; rows/cols 1-based, starting at (1,1)):

1. Per row, per cell: run the **occupancy check** first — if the current slot is covered by a previous merge (openpyxl's `MergedCell`), advance the column pointer until a free slot;
2. colspan/rowspan attributes go through the official `string_to_int`: only `isdigit` converts, **anything else is 0** (`"abc"`/`"0"`/`""` → 0; a missing attribute (None) defaults to "1" per the official `get(attr, "1")`). Note Python's `isdigit` is also True for fullwidth digits, so `int("２") = 2`;
3. Allocate (row, col); if rowspan>1 or colspan>1 a merge fires over `[row..row+rowspan-1] × [col..col+colspan-1]` (closed), the anchor being (row, col) itself, all other covered slots marked occupied;
4. Column pointer advances by `col += colspan` — a 0-parsed colspan does **not** advance, so the next cell **overwrites the same slot** (official quirk);
5. At row end the column pointer resets to 1 and the row pointer increments;
6. `grid`: each slot stores the last cell_id allocated to it; non-anchor slots inside a merge are None (openpyxl `MergedCell` empty-value storage semantics); `num_rows` extends by merge bottoms — openpyxl's `merge_cells` auto-expands the sheet, so a `rowspan` past the last row grows the grid.

Returns `{"cells": [[cell_id, row, col, text], ...], "merges": [[r1,c1,r2,c2], ...], "num_rows", "num_cols", "grid"}`.

## Constraints

- 0 ≤ rows ≤ 100, 0 ≤ cells per row ≤ 50, attribute strings ≤ 8 chars.
- rows not a list, cell not a 3-tuple, text not str, attribute neither str nor None → `ValueError`.
