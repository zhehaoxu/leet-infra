def _string_to_int(s):
    if s.isdigit():
        return int(s)
    return 0


def solve(rows):
    if not isinstance(rows, (list, tuple)):
        raise ValueError("rows must be a list of table rows")
    for tr in rows:
        if not isinstance(tr, (list, tuple)):
            raise ValueError("each row must be a list of cells")
        for td in tr:
            if not isinstance(td, (list, tuple)) or len(td) != 3:
                raise ValueError("each cell must be [text, colspan, rowspan]")
            text, cs_attr, rs_attr = td
            if not isinstance(text, str):
                raise ValueError("cell text must be a string")
            for attr in (cs_attr, rs_attr):
                if attr is not None and not isinstance(attr, str):
                    raise ValueError("colspan/rowspan must be str or None")

    occupied = set()
    merges = []
    cells = []
    col = 1
    row = 1
    cid = 0
    for tr in rows:
        for td in tr:
            text, cs_attr, rs_attr = td
            cs = _string_to_int("1" if cs_attr is None else cs_attr)
            rs = _string_to_int("1" if rs_attr is None else rs_attr)
            while (row, col) in occupied:
                col += 1
            cells.append([cid, row, col, text])
            if rs > 1 or cs > 1:
                merges.append([row, col, row + rs - 1, col + cs - 1])
                for rr in range(row, row + rs):
                    for cc in range(col, col + cs):
                        if (rr, cc) != (row, col):
                            occupied.add((rr, cc))
            col += cs
            cid += 1
        row += 1
        col = 1

    num_cols = 0
    for _, r, c, _ in cells:
        num_cols = max(num_cols, c)
    grid_rows = len(rows)
    for r1, c1, r2, c2 in merges:
        num_cols = max(num_cols, c2)
        # openpyxl merge_cells 会自动扩展 sheet：merge 右端越界行也计入
        grid_rows = max(grid_rows, r2)
    grid = [[None] * num_cols for _ in range(grid_rows)]
    for cid, r, c, _t in cells:
        grid[r - 1][c - 1] = cid
    for r1, c1, r2, c2 in merges:
        for rr in range(r1, r2 + 1):
            for cc in range(c1, c2 + 1):
                if (rr, cc) != (r1, c1):
                    grid[rr - 1][cc - 1] = None
    return {
        "cells": cells,
        "merges": merges,
        "num_rows": grid_rows,
        "num_cols": num_cols,
        "grid": grid,
    }
