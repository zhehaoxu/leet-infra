GRID_W = 1001  # 0..1000 共 1001 列


def solve(boxes, page_w, page_h):
    def _page_ok(v, name):
        if isinstance(v, bool) or not isinstance(v, (int, float)) \
                or isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
            raise ValueError(f"{name} must be a finite number")
        if v <= 0:
            raise ValueError(f"{name} must be positive")

    _page_ok(page_w, "page_w")
    _page_ok(page_h, "page_h")
    if not isinstance(boxes, (list, tuple)) or len(boxes) == 0:
        raise ValueError("boxes must be a non-empty list")

    def norm(v, page):
        n = round(1000.0 * v / page)
        if n < 0:
            n = 0
        elif n > 1000:
            n = 1000
        return n

    norm_boxes = []
    wh_ids = []
    corner_grid = []
    center_grid = []
    for b in boxes:
        if not isinstance(b, (list, tuple)) or len(b) != 4:
            raise ValueError("each box must be [x0, y0, x1, y1]")
        for x in b:
            if isinstance(x, bool) or not isinstance(x, (int, float)) \
                    or isinstance(x, float) and (x != x or x in (float("inf"), float("-inf"))):
                raise ValueError("box entries must be finite numbers")
        x0, y0, x1, y1 = (norm(b[0], page_w), norm(b[1], page_h),
                          norm(b[2], page_w), norm(b[3], page_h))
        if x0 > x1 or y0 > y1:
            raise ValueError("normalized box must satisfy x0<=x1 and y0<=y1")
        norm_boxes.append([x0, y0, x1, y1])
        wh_ids.append([x1 - x0, y1 - y0])
        g = lambda x, y: y * GRID_W + x
        corner_grid.append([g(x0, y0), g(x1, y0), g(x0, y1), g(x1, y1)])
        center_grid.append(g((x0 + x1) // 2, (y0 + y1) // 2))
    return {
        "norm_boxes": norm_boxes,
        "wh_ids": wh_ids,
        "corner_grid": corner_grid,
        "center_grid": center_grid,
    }
