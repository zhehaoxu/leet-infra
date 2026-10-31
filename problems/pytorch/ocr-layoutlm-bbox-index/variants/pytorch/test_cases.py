GRID_W = 1001


def _py(boxes, page_w, page_h):
    """oracle：独立实现（整除展开 norm、divmod 求网格坐标）。"""
    norm_boxes = []
    wh_ids = []
    corner_grid = []
    center_grid = []
    for b in boxes:
        row = []
        for i, v in enumerate(b):
            n = int(round(1000.0 * v / (page_w if i % 2 == 0 else page_h)))
            row.append(min(1000, max(0, n)))
        x0, y0, x1, y1 = row
        norm_boxes.append(row)
        wh_ids.append([x1 - x0, y1 - y0])
        quad = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]
        corner_grid.append([y * GRID_W + x for x, y in quad])
        cx, cy = divmod((x0 + x1), 2)[0], (y0 + y1) // 2
        center_grid.append(cy * GRID_W + cx)
    return {"norm_boxes": norm_boxes, "wh_ids": wh_ids,
            "corner_grid": corner_grid, "center_grid": center_grid}


def _compare(solve, boxes, page_w, page_h):
    led = solve(boxes, page_w, page_h)
    ref = _py(boxes, page_w, page_h)
    assert led == ref, f"solve vs oracle:\n{led}\nvs\n{ref}"
    return led


def check_anchor_basic(solve):
    # page 1000x500：x 原样、y 翻倍
    led = _compare(solve, [[10, 20, 110, 220]], 1000, 500)
    assert led["norm_boxes"] == [[10, 40, 110, 440]]
    assert led["wh_ids"] == [[100, 400]]
    g = lambda x, y: y * 1001 + x
    assert led["corner_grid"] == [[g(10, 40), g(110, 40), g(10, 440), g(110, 440)]]
    assert led["center_grid"] == [g(60, 240)]


def check_clamp(solve):
    # 超页与负坐标 clamp 到 [0, 1000]
    led = _compare(solve, [[-50, -1, 5000, 900]], 1000, 1000)
    assert led["norm_boxes"] == [[0, 0, 1000, 900]]
    assert led["wh_ids"] == [[1000, 900]]
    # 恰好 1000 不截断
    led2 = _compare(solve, [[1000, 0, 1000, 1000]], 1000, 1000)
    assert led2["norm_boxes"] == [[1000, 0, 1000, 1000]]


def check_rounding(solve):
    # 四舍五入：page 3 → 1/3 → 333.33→333、2/3 → 666.67→667
    led = _compare(solve, [[1, 2, 3, 3]], 3, 3)
    assert led["norm_boxes"] == [[333, 667, 1000, 1000]]
    # banker's rounding：page_w=2000、x=3 → 1.5 → round 到偶数 2（int 截断会得 1）
    led2 = _compare(solve, [[3, 3, 5, 5]], 2000, 2000)
    assert led2["norm_boxes"] == [[2, 2, 2, 2]] or led2["norm_boxes"][0][0] == 2
    led3 = _compare(solve, [[1, 1, 1, 1]], 2000, 2000)  # 0.5 → 0（round 到偶）
    assert led3["norm_boxes"] == [[0, 0, 0, 0]]


def check_full_page_and_degenerate(solve):
    # 全页 box；退化为线/点的 box（w 或 h 为 0）
    led = _compare(solve, [[0, 0, 1000, 1000]], 1000, 1000)
    assert led["wh_ids"] == [[1000, 1000]]
    assert led["corner_grid"][0] == [0, 1000, 1001 * 1000, 1001 * 1000 + 1000]
    led2 = _compare(solve, [[5, 5, 5, 5]], 1000, 1000)
    assert led2["wh_ids"] == [[0, 0]]
    assert led2["center_grid"] == [5 * 1001 + 5]


def check_multi_box(solve):
    led = _compare(solve, [[0, 0, 100, 100], [500, 500, 600, 600]], 1000, 1000)
    assert led["norm_boxes"] == [[0, 0, 100, 100], [500, 500, 600, 600]]
    assert led["center_grid"] == [50 * 1001 + 50, 550 * 1001 + 550]
    assert led["wh_ids"] == [[100, 100], [100, 100]]


def check_invalid_inputs(solve):
    def ve(*args):
        try:
            solve(*args)
        except ValueError:
            return
        raise AssertionError("expected ValueError")
    ve([], 1000, 1000)                       # 空 boxes
    ve([[[1, 2, 3]]], 1000, 1000)            # 非 4 元
    ve([[["a", 2, 3, 4]]], 1000, 1000)       # 非数
    ve([[[1, 2, 3, True]]], 1000, 1000)      # bool
    ve([[[1, 2, 3, float("nan")]]], 1000, 1000)
    ve([[[100, 0, 50, 100]]], 1000, 1000)    # x0 > x1（归一化后仍如此）
    ve([[[0, 100, 100, 50]]], 1000, 1000)    # y0 > y1
    ve([[[0, 0, 10, 10]]], 0, 1000)          # page_w 非正
    ve([[[0, 0, 10, 10]]], 1000, 0)
    ve([[[0, 0, 10, 10]]], -5, 1000)
    ve([[[0, 0, 10, 10]]], True, 1000)       # page bool


def check_oracle_random(solve):
    import random
    rng = random.Random(20260669)
    for _ in range(150):
        pw = rng.choice([100, 612, 794, 1000, 1275, 2000, 3508])
        ph = rng.choice([100, 794, 1000, 1123, 1400, 2000, 2480])
        n = rng.randint(1, 8)
        boxes = []
        for _ in range(n):
            x0 = rng.randint(0, pw)
            y0 = rng.randint(0, ph)
            x1 = rng.randint(x0, pw)
            y1 = rng.randint(y0, ph)
            boxes.append([x0, y0, x1, y1])
        led = solve(boxes, pw, ph)
        ref = _py(boxes, pw, ph)
        assert led == ref, "random mismatch"
