def _py(rows):
    """oracle：merge 优先于分配的独立实现——先把每个 cell 的目标格声明出来再回填 grid。

    与 solve 的「顺序扫描 + 占用集合」路径不同：oracle 用逐步推进的
    显式指针循环 + 字典占用表，最后独立重算 grid（覆盖优先、再清非 anchor）。
    """
    if not isinstance(rows, (list, tuple)):
        raise ValueError
    for tr in rows:
        for td in tr:
            if not isinstance(td, (list, tuple)) or len(td) != 3:
                raise ValueError
            if not isinstance(td[0], str):
                raise ValueError
            for attr in (td[1], td[2]):
                if attr is not None and not isinstance(attr, str):
                    raise ValueError

    def s2i(s):
        # 同一官方规则，但走 chr-by-chr 判定路径复刻 isdigit
        if s and all(ch.isdigit() for ch in s):
            return int(s)
        return 0

    occ = {}
    merges = []
    cells = []
    col = 1
    row = 1
    cid = 0
    for tr in rows:
        col = 1
        for td in tr:
            cs = s2i(td[1]) if td[1] is not None else 1
            rs = s2i(td[2]) if td[2] is not None else 1
            while occ.get((row, col), False):
                col += 1
            cells.append([cid, row, col, td[0]])
            if rs > 1 or cs > 1:
                merges.append([row, col, row + rs - 1, col + cs - 1])
                rr = row
                while rr < row + rs:
                    cc = col
                    while cc < col + cs:
                        if (rr, cc) != (row, col):
                            occ[(rr, cc)] = True
                        cc += 1
                    rr += 1
            col = col + cs
            cid += 1
        row += 1

    num_cols = 0
    for entry in cells:
        num_cols = max(num_cols, entry[2])
    grid_rows = len(rows)
    for m in merges:
        num_cols = max(num_cols, m[3])
        grid_rows = max(grid_rows, m[2])
    grid = []
    for _r in range(grid_rows):
        grid.append([None] * num_cols)
    for entry in cells:
        grid[entry[1] - 1][entry[2] - 1] = entry[0]
    for m in merges:
        rr = m[0]
        while rr <= m[2]:
            cc = m[1]
            while cc <= m[3]:
                if not (rr == m[0] and cc == m[1]):
                    grid[rr - 1][cc - 1] = None
                cc += 1
            rr += 1
    return {
        "cells": cells,
        "merges": merges,
        "num_rows": grid_rows,
        "num_cols": num_cols,
        "grid": grid,
    }


def _compare(solve, rows):
    led = solve(rows)
    ref = _py(rows)
    assert led == ref, f"solve vs oracle:\n{led}\nvs\n{ref}"
    return led


def check_anchor_header_colspan(solve):
    # h1 colspan=2 → merge [1,1,1,2]，h2 落 (1,3)；grid [0,None,1]
    led = _compare(solve, [[["h1", "2", None], ["h2", None, None]]])
    assert led["cells"] == [[0, 1, 1, "h1"], [1, 1, 3, "h2"]]
    assert led["merges"] == [[1, 1, 1, 2]]
    assert led["num_cols"] == 3
    assert led["grid"] == [[0, None, 1]]


def check_rowspan_skip(solve):
    # a rowspan=2 占 (1,1)/(2,1)；第 2 行的 b 从 (2,2) 开始（占用跳过）
    led = _compare(
        solve,
        [[["a", None, "2"], ["x", None, None]], [["b", None, None], ["c", None, None]]],
    )
    assert led["cells"] == [[0, 1, 1, "a"], [1, 1, 2, "x"], [2, 2, 2, "b"], [3, 2, 3, "c"]]
    assert led["merges"] == [[1, 1, 2, 1]]
    assert led["grid"] == [[0, 1, None], [None, 2, 3]]


def check_non_numeric_attr_zero(solve):
    # 官方 string_to_int：非数字属性 → 0 → 不推进列，下一 cell 覆盖同一格
    led = _compare(solve, [[["a", "abc", None], ["b", None, None], ["c", None, None]]])
    assert led["cells"] == [[0, 1, 1, "a"], [1, 1, 1, "b"], [2, 1, 2, "c"]]
    assert led["merges"] == []
    # "0" 同理
    led2 = _compare(solve, [[["a", "0", None], ["b", None, None]]])
    assert led2["cells"] == [[0, 1, 1, "a"], [1, 1, 1, "b"]]


def check_fullwidth_digit(solve):
    # Python isdigit 对全角数字也为 True → int("２") = 2（官方路径真实行为）
    led = _compare(solve, [[["h", "２", None], ["x", None, None]]])
    assert led["cells"] == [[0, 1, 1, "h"], [1, 1, 3, "x"]]
    assert led["merges"] == [[1, 1, 1, 2]]


def check_both_span_rect(solve):
    # rowspan=2 × colspan=2 的矩形 merge，随后两行都要跳过被占格
    led = _compare(
        solve,
        [
            [["big", "2", "2"], ["u", None, None], ["v", None, None]],
            [["w", None, None], ["z", None, None]],
        ],
    )
    # big (1,1) merge [1,1,2,2]；u (1,3)；v (1,4)
    # 第 2 行：(2,1)/(2,2) 被占 → w (2,3)、z (2,4)
    assert led["cells"] == [
        [0, 1, 1, "big"], [1, 1, 3, "u"], [2, 1, 4, "v"], [3, 2, 3, "w"], [4, 2, 4, "z"],
    ]
    assert led["merges"] == [[1, 1, 2, 2]]
    assert led["grid"] == [
        [0, None, 1, 2],
        [None, None, 3, 4],
    ]


def check_empty_and_defaults(solve):
    # 空表；单空行；属性全缺省
    led = _compare(solve, [])
    assert led == {"cells": [], "merges": [], "num_rows": 0, "num_cols": 0, "grid": []}
    led2 = _compare(solve, [[]])
    assert led2["num_rows"] == 1 and led2["num_cols"] == 0 and led2["grid"] == [[]]
    led3 = _compare(solve, [[["a", None, None], ["b", None, None]]])
    assert led3["cells"] == [[0, 1, 1, "a"], [1, 1, 2, "b"]]
    assert led3["grid"] == [[0, 1]]


def check_late_row_widens(solve):
    # 后行更宽：num_cols 取全表最大占用列；且 merge 右端也参与宽度
    led = _compare(
        solve,
        [[["a", None, None]], [["b", "3", None], ["c", None, None]]],
    )
    assert led["num_cols"] == 4
    assert led["grid"] == [[0, None, None, None], [1, None, None, 2]]
    assert led["merges"] == [[2, 1, 2, 3]]


def check_merge_extends_rows(solve):
    # rowspan 越过末行：openpyxl merge_cells 自动扩展 sheet，num_rows 按 merge 右端扩展
    led = _compare(solve, [[["a", None, "3"], ["x", None, None]]])
    assert led["num_rows"] == 3
    assert led["merges"] == [[1, 1, 3, 1]]
    assert led["grid"] == [[0, 1], [None, None], [None, None]]


def check_invalid_inputs(solve):
    def ve(rows):
        try:
            solve(rows)
        except ValueError:
            return
        raise AssertionError("expected ValueError")

    ve("nope")
    ve([["not-a-td"]])
    ve([[["a", "1"]]])               # 缺 rowspan 位
    ve([[[1, "1", None]]])           # text 非 str
    ve([[["a", 2, None]]])           # colspan 非 str/None
    ve([[["a", None, 3]]])           # rowspan 非 str/None
    ve([["ab", "1", None]])          # 行非嵌套


def check_oracle_random(solve):
    import random

    rng = random.Random(20260689)
    attrs = [None, "1", "2", "3", "0", "abc", "２", ""]
    for _ in range(200):
        nr = rng.randint(0, 8)
        rows = []
        for _r in range(nr):
            nc = rng.randint(0, 5)
            tr = []
            for _c in range(nc):
                tr.append(
                    [
                        f"t{rng.randint(0, 99)}",
                        rng.choice(attrs),
                        rng.choice(attrs),
                    ]
                )
            rows.append(tr)
        led = solve(rows)
        ref = _py(rows)
        assert led == ref, "random mismatch"
