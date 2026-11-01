def solve(bboxes, w):
    """PP-Structure recovery `sorted_layout_boxes` 双栏阅读顺序账。

    返回 {"order": [原始下标...], "layouts": ["single"/"double"...]}。
    """
    n = len(bboxes)
    if n == 1:
        return {"order": [0], "layouts": ["single"]}

    # 官方 sorted(res, key=(y0, x0))：Python 稳定排序，键相同保持原始顺序
    scan = sorted(range(n), key=lambda i: (bboxes[i][1], bboxes[i][0]))

    order = []
    layouts = []
    res_left = []
    res_right = []
    i = 0

    while True:
        if i >= n:
            break
        if i == n - 1:
            cur = bboxes[scan[i]]
            prev = bboxes[scan[i - 1]]
            if cur[1] > prev[3] and cur[0] < w / 2 and cur[2] > w / 2:
                order.extend(res_left)
                layouts.extend(["double"] * len(res_left))
                order.extend(res_right)
                layouts.extend(["double"] * len(res_right))
                order.append(scan[i])
                layouts.append("single")
            else:
                if cur[2] > w / 2:
                    res_right.append(scan[i])
                    order.extend(res_left)
                    layouts.extend(["double"] * len(res_left))
                    order.extend(res_right)
                    layouts.extend(["double"] * len(res_right))
                elif cur[0] < w / 2:
                    res_left.append(scan[i])
                    order.extend(res_left)
                    layouts.extend(["double"] * len(res_left))
                    order.extend(res_right)
                    layouts.extend(["double"] * len(res_right))
                # 否则：退化零宽框（x0 == x1 == w/2）连同桶一并丢弃
            res_left = []
            res_right = []
            break
        else:
            cur = bboxes[scan[i]]
            if cur[0] < w / 4 and cur[2] < 3 * w / 4:
                res_left.append(scan[i])
                i += 1
            elif cur[0] > w / 4 and cur[2] > w / 2:
                res_right.append(scan[i])
                i += 1
            else:
                order.extend(res_left)
                layouts.extend(["double"] * len(res_left))
                order.extend(res_right)
                layouts.extend(["double"] * len(res_right))
                order.append(scan[i])
                layouts.append("single")
                res_left = []
                res_right = []
                i += 1

    return {"order": order, "layouts": layouts}
