"""order 836 判题用例：独立 oracle 用纯 Python 重写，不 import solution。

4 个 check_*：overlap_gate / interval_ledger / line_grouping / det_box_merge。
每个都做全网格对拍 + 手算锚点（手算锚点独立于 oracle 与 solution）。
"""

SPECIAL = {"interline_equation", "image", "table"}


# ---------------------------------------------------------------- oracle
def ora_overlap(box_a, box_b, axis, thr):
    a, b = list(box_a), list(box_b)
    if axis == "y":
        lo = max(a[1], b[1]); hi = min(a[3], b[3])
        ext_a = a[3] - a[1]; ext_b = b[3] - b[1]
    else:
        lo = max(a[0], b[0]); hi = min(a[2], b[2])
        ext_a = a[2] - a[0]; ext_b = b[2] - b[0]
    ov = hi - lo
    if ov < 0:
        ov = 0
    ext = ext_a if ext_a < ext_b else ext_b
    if ext <= 0:
        return {"overlap": ov, "min_extent": ext, "ratio": None,
                "exceeds": False, "zero_extent": True}
    r = ov / ext
    return {"overlap": ov, "min_extent": ext, "ratio": r,
            "exceeds": r > thr, "zero_extent": False}


def ora_merge(ivs):
    seq = sorted(ivs, key=lambda p: p[0])
    out = []
    for s, e in seq:
        if out and out[-1][1] >= s:
            if e > out[-1][1]:
                out[-1][1] = e
        else:
            out.append([s, e])
    return out


def ora_remove(orig, masks):
    s0, e0 = orig
    segs = []
    ptr = s0
    for ms, me in ora_merge(masks):
        if ms > e0 or me < s0:
            continue
        if ptr < ms:
            segs.append([ptr, ms - 1])
        if me + 1 > ptr:
            ptr = me + 1
    if ptr <= e0:
        segs.append([ptr, e0])
    return segs


def ora_interval(intervals, masks, original):
    mi = ora_merge(intervals)
    mm = ora_merge(masks)
    if original is None:
        rem, cov, pts = [], 0, 0
    else:
        rem = ora_remove(original, masks)
        total = max(0, original[1] - original[0] + 1)
        pts = 0
        for s, e in rem:
            pts += max(0, e - s + 1)
        cov = total - pts
    return {"merged_intervals": mi, "merged_masks": mm, "remaining": rem,
            "n_segments": len(rem), "covered_points": cov,
            "remaining_points": pts}


def ora_span_ratio(b):
    w = b[2] - b[0]; h = b[3] - b[1]
    if w <= 0 or h <= 0:
        return None
    return h / w


def _bbox_from_boxes(boxes):
    x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes); y1 = max(b[3] for b in boxes)
    return [x0, y0, x1, y1]


def ora_group(spans, thr):
    items = [{"bbox": list(s["bbox"]), "type": s["type"]} for s in spans]
    down = 0
    for it in items:
        if it["type"] == "interline_equation":
            it["type"] = "inline_equation"
            down += 1
    valid = 0; vert = 0
    for it in items:
        b = it["bbox"]
        if not b or len(b) < 4:
            continue
        r = ora_span_ratio(b)
        if r is None:
            continue
        valid += 1
        if r > 2:
            vert += 1
    is_vert = (valid > 0) and (vert / valid > 0.8)
    if is_vert:
        seq = sorted(items, key=lambda s: s["bbox"][2], reverse=True)
        axis, inner, orient = "x", 1, "vertical"
    else:
        seq = sorted(items, key=lambda s: s["bbox"][1])
        axis, inner, orient = "y", 0, "horizontal"
    groups = []
    if seq:
        cur = [seq[0]]
        for it in seq[1:]:
            if it["type"] in SPECIAL or any(x["type"] in SPECIAL for x in cur):
                groups.append(cur); cur = [it]
            elif ora_overlap(it["bbox"], cur[-1]["bbox"], axis, thr)["exceeds"]:
                cur.append(it)
            else:
                groups.append(cur); cur = [it]
        groups.append(cur)
    for g in groups:
        g.sort(key=lambda s: s["bbox"][inner])
    return {"vertical": is_vert, "valid_span_count": valid,
            "vertical_span_count": vert, "downgraded": down,
            "lines": [[list(s["bbox"]) for s in g] for g in groups],
            "line_bboxes": [_bbox_from_boxes([s["bbox"] for s in g])
                            for g in groups],
            "orientation": orient}


def ora_angle(poly):
    p1, p2, p3, p4 = poly
    hh = ((p4[1] - p1[1]) + (p3[1] - p2[1])) / 2
    dy = p3[1] - p1[1]
    return not (0.8 * hh <= dy <= 1.2 * hh)


def ora_p2b(poly):
    return [poly[0][0], poly[0][1], poly[1][0], poly[2][1]]


def ora_merge_spans(boxes):
    seq = sorted(boxes, key=lambda b: b[0])
    out = []
    for b in seq:
        if not out or out[-1][2] < b[0]:
            out.append(list(b))
        else:
            prev = out.pop()
            out.append([min(prev[0], b[0]), min(prev[1], b[1]),
                        max(prev[2], b[2]), max(prev[3], b[3])])
    return out


def ora_det(polys, ratio_thr, ov_thr):
    flags, horiz, ang = [], [], []
    for p in polys:
        if ora_angle(p):
            flags.append(True); ang.append(p)
        else:
            flags.append(False); horiz.append(ora_p2b(p))
    seq = sorted(horiz, key=lambda b: b[1])
    lines = []
    if seq:
        cur = [seq[0]]
        for b in seq[1:]:
            if ora_overlap(b, cur[-1], "y", ov_thr)["exceeds"]:
                cur.append(b)
            else:
                lines.append(cur); cur = [b]
        lines.append(cur)
    lbs, trig, result = [], [], []
    for line in lines:
        lb = _bbox_from_boxes(line)
        lbs.append(lb)
        t = (lb[2] - lb[0]) > (lb[3] - lb[1]) * ratio_thr
        trig.append(t)
        result.extend(ora_merge_spans(line) if t else [list(b) for b in line])
    result.extend(ora_p2b(p) for p in ang)
    return {"angle_flags": flags, "horizontal_bboxes": [list(b) for b in horiz],
            "line_bboxes": lbs, "ratio_triggered": trig,
            "result_bboxes": [list(b) for b in result]}


# ---------------------------------------------------------------- checks
def check_overlap_gate(solve):
    pairs = [
        ([0, 0, 10, 10], [0, 5, 10, 15]),
        ([0, 0, 10, 10], [0, 2, 10, 12]),
        ([0, 0, 10, 10], [0, 0, 10, 0]),
        ([0, 0, 10, 10], [3, 3, 7, 13]),
        ([0, 0, 4, 20], [0, 0, 20, 4]),
        ([5, 5, 15, 15], [5, 5, 15, 15]),
        ([0, 0, 10, 10], [20, 0, 30, 10]),
        ([0, 0, 10, 10], [0, 20, 10, 30]),
    ]
    n = 0
    for a, b in pairs:
        for axis in ("y", "x"):
            for thr in (0.5, 0.6, 0.8, 1.0):
                got = solve(mode="overlap_gate", bbox1=a, bbox2=b,
                            axis=axis, threshold=thr)
                exp = ora_overlap(a, b, axis, thr)
                assert got == exp, ("gate a=%r b=%r axis=%r thr=%r\n got=%r\n exp=%r"
                                    % (a, b, axis, thr, got, exp))
                n += 1
    # 手算锚点 1：ratio=0.5 不超 0.8
    assert solve(mode="overlap_gate", bbox1=[0, 0, 10, 10],
                 bbox2=[0, 5, 10, 15], axis="y", threshold=0.8) == {
        "overlap": 5, "min_extent": 10, "ratio": 0.5,
        "exceeds": False, "zero_extent": False}
    # 手算锚点 2：ratio 恰等于阈值，严格 > 必须为 False
    assert solve(mode="overlap_gate", bbox1=[0, 0, 10, 10],
                 bbox2=[0, 2, 10, 12], axis="y", threshold=0.8) == {
        "overlap": 8, "min_extent": 10, "ratio": 0.8,
        "exceeds": False, "zero_extent": False}
    # 手算锚点 3：退化框高为 0，除零保护
    assert solve(mode="overlap_gate", bbox1=[0, 0, 10, 0],
                 bbox2=[0, 0, 10, 10], axis="y", threshold=0.6) == {
        "overlap": 0, "min_extent": 0, "ratio": None,
        "exceeds": False, "zero_extent": True}
    # 手算锚点 4：y 轴完全分离时负 overlap 必须钳位为 0
    assert solve(mode="overlap_gate", bbox1=[0, 0, 10, 10],
                 bbox2=[0, 20, 10, 30], axis="y", threshold=0.6) == {
        "overlap": 0, "min_extent": 10, "ratio": 0.0,
        "exceeds": False, "zero_extent": False}
    print("overlap_gate: %d grid + 4 anchors OK" % n)


def check_interval_ledger(solve):
    ivs = [
        [[0, 10], [10, 20], [25, 30]],
        [[5, 8], [0, 3], [7, 12]],
        [[0, 5]],
        [],
        [[9, 9], [9, 9], [3, 4]],
    ]
    masks = [
        [],
        [[2, 4], [6, 9]],
        [[20, 30], [0, 10]],
        [[1, 3], [2, 5], [8, 9]],
    ]
    originals = [None, [0, 12], [0, 10], [0, 30]]
    n = 0
    for iv in ivs:
        for mk in masks:
            for orig in originals:
                got = solve(mode="interval_ledger", intervals=iv,
                            masks=mk, original=orig)
                exp = ora_interval(iv, mk, orig)
                assert got == exp, ("interval iv=%r mk=%r orig=%r\n got=%r\n exp=%r"
                                    % (iv, mk, orig, got, exp))
                n += 1
    # 手算锚点 1：相邻闭区间被合并（严格 <）
    assert solve(mode="interval_ledger", intervals=[[0, 10], [10, 20], [25, 30]],
                 masks=[], original=None) == {
        "merged_intervals": [[0, 20], [25, 30]], "merged_masks": [],
        "remaining": [], "n_segments": 0, "covered_points": 0,
        "remaining_points": 0}
    # 手算锚点 2：±1 离散裁除
    assert solve(mode="interval_ledger", intervals=[[5, 8], [0, 3], [7, 12]],
                 masks=[[2, 4], [6, 9]], original=[0, 12]) == {
        "merged_intervals": [[0, 3], [5, 12]],
        "merged_masks": [[2, 4], [6, 9]],
        "remaining": [[0, 1], [5, 5], [10, 12]], "n_segments": 3,
        "covered_points": 7, "remaining_points": 6}
    # 手算锚点 3：mask 覆盖全部 + 外部 mask 跳过
    assert solve(mode="interval_ledger", intervals=[[0, 5]],
                 masks=[[20, 30], [0, 10]], original=[0, 10]) == {
        "merged_intervals": [[0, 5]], "merged_masks": [[0, 10], [20, 30]],
        "remaining": [], "n_segments": 0, "covered_points": 11,
        "remaining_points": 0}
    print("interval_ledger: %d grid + 3 anchors OK" % n)


def check_line_grouping(solve):
    sets = [
        [],
        [{"bbox": [0, 0, 10, 10], "type": "text"}],
        [{"bbox": [0, 0, 10, 10], "type": "text"},
         {"bbox": [12, 1, 20, 9], "type": "text"},
         {"bbox": [0, 20, 10, 30], "type": "text"}],
        [{"bbox": [0, 0, 10, 10], "type": "text"},
         {"bbox": [12, 1, 20, 9], "type": "interline_equation"},
         {"bbox": [22, 2, 30, 8], "type": "text"}],
        [{"bbox": [100, 0, 110, 30], "type": "text"},
         {"bbox": [90, 0, 100, 30], "type": "text"},
         {"bbox": [80, 0, 90, 30], "type": "image"}],
        [{"bbox": [0, 0, 10, 10], "type": "image"},
         {"bbox": [12, 1, 20, 9], "type": "text"},
         {"bbox": [22, 2, 30, 8], "type": "text"}],
        [{"bbox": [0, 0, 10, 0], "type": "text"},
         {"bbox": [0, 0, 4, 20], "type": "text"},
         {"bbox": [5, 0, 8, 20], "type": "table"}],
        # 4 个 span 的 h/w 恰为 2（竖排门限严格 >）：正确 0 个竖排，变异 4 个竖排
        [{"bbox": [0, 0, 10, 20], "type": "text"},
         {"bbox": [12, 0, 22, 20], "type": "text"},
         {"bbox": [24, 0, 34, 20], "type": "text"},
         {"bbox": [36, 0, 46, 20], "type": "text"}],
        # 竖排占比恰 4/5=0.8（dominance 严格 >）
        [{"bbox": [0, 0, 10, 30], "type": "text"},
         {"bbox": [12, 0, 22, 30], "type": "text"},
         {"bbox": [24, 0, 34, 30], "type": "text"},
         {"bbox": [36, 0, 46, 30], "type": "text"},
         {"bbox": [48, 0, 58, 10], "type": "text"}],
        # 行内 x0 顺序与 y0 顺序相反，钉死 inner_key = bbox[0]
        [{"bbox": [10, 0, 20, 10], "type": "text"},
         {"bbox": [0, 1, 8, 9], "type": "text"}],
    ]
    n = 0
    for sp in sets:
        for thr in (0.6, 0.8):
            got = solve(mode="line_grouping", spans=sp, threshold=thr)
            exp = ora_group(sp, thr)
            assert got == exp, ("group sp=%r thr=%r\n got=%r\n exp=%r"
                                % (sp, thr, got, exp))
            n += 1
    # 手算锚点 1：横排两行 + 外接矩形
    assert solve(mode="line_grouping", spans=[
        {"bbox": [0, 0, 10, 10], "type": "text"},
        {"bbox": [12, 1, 20, 9], "type": "text"},
        {"bbox": [0, 20, 10, 30], "type": "text"}], threshold=0.6) == {
        "vertical": False, "valid_span_count": 3, "vertical_span_count": 0,
        "downgraded": 0,
        "lines": [[[0, 0, 10, 10], [12, 1, 20, 9]], [[0, 20, 10, 30]]],
        "line_bboxes": [[0, 0, 20, 10], [0, 20, 10, 30]],
        "orientation": "horizontal"}
    # 手算锚点 2：INTERLINE 降级后不再触发 special 断行，三段合成一行
    assert solve(mode="line_grouping", spans=[
        {"bbox": [0, 0, 10, 10], "type": "text"},
        {"bbox": [12, 1, 20, 9], "type": "interline_equation"},
        {"bbox": [22, 2, 30, 8], "type": "text"}], threshold=0.6) == {
        "vertical": False, "valid_span_count": 3, "vertical_span_count": 0,
        "downgraded": 1,
        "lines": [[[0, 0, 10, 10], [12, 1, 20, 9], [22, 2, 30, 8]]],
        "line_bboxes": [[0, 0, 30, 10]], "orientation": "horizontal"}
    # 手算锚点 3：竖排判定（3/3 > 0.8）+ 按 bbox[2] 降序 + image 强制断行
    assert solve(mode="line_grouping", spans=[
        {"bbox": [100, 0, 110, 30], "type": "text"},
        {"bbox": [90, 0, 100, 30], "type": "text"},
        {"bbox": [80, 0, 90, 30], "type": "image"}], threshold=0.6) == {
        "vertical": True, "valid_span_count": 3, "vertical_span_count": 3,
        "downgraded": 0,
        "lines": [[[100, 0, 110, 30]], [[90, 0, 100, 30]],
                  [[80, 0, 90, 30]]],
        "line_bboxes": [[100, 0, 110, 30], [90, 0, 100, 30],
                        [80, 0, 90, 30]],
        "orientation": "vertical"}
    # 手算锚点 4：不传 threshold 时默认 0.6 生效（0.7 重叠判同行）
    assert solve(mode="line_grouping", spans=[
        {"bbox": [0, 0, 10, 10], "type": "text"},
        {"bbox": [12, 3, 20, 13], "type": "text"}]) == {
        "vertical": False, "valid_span_count": 2, "vertical_span_count": 0,
        "downgraded": 0,
        "lines": [[[0, 0, 10, 10], [12, 3, 20, 13]]],
        "line_bboxes": [[0, 0, 20, 13]], "orientation": "horizontal"}
    # 手算锚点 5：行内按 x0 排序（与 y0 顺序相反）
    assert solve(mode="line_grouping", spans=[
        {"bbox": [10, 0, 20, 10], "type": "text"},
        {"bbox": [0, 1, 8, 9], "type": "text"}], threshold=0.6)["lines"] == [
        [[0, 1, 8, 9], [10, 0, 20, 10]]]
    # 手算锚点 6：h/w 恰为 2 不算竖排（严格 >），4 个全为 2 时 is_vert 仍 False
    r6 = solve(mode="line_grouping", spans=[
        {"bbox": [0, 0, 10, 20], "type": "text"},
        {"bbox": [12, 0, 22, 20], "type": "text"},
        {"bbox": [24, 0, 34, 20], "type": "text"},
        {"bbox": [36, 0, 46, 20], "type": "text"}], threshold=0.6)
    assert r6["valid_span_count"] == 4 and r6["vertical_span_count"] == 0
    assert r6["vertical"] is False
    # 手算锚点 7：竖排占比恰 0.8，严格 > 不判竖排
    r7 = solve(mode="line_grouping", spans=[
        {"bbox": [0, 0, 10, 30], "type": "text"},
        {"bbox": [12, 0, 22, 30], "type": "text"},
        {"bbox": [24, 0, 34, 30], "type": "text"},
        {"bbox": [36, 0, 46, 30], "type": "text"},
        {"bbox": [48, 0, 58, 10], "type": "text"}], threshold=0.6)
    assert r7["valid_span_count"] == 5 and r7["vertical_span_count"] == 4
    assert r7["vertical"] is False
    print("line_grouping: %d grid + 7 anchors OK" % n)


def check_det_box_merge(solve):
    def rect(x0, y0, x1, y1):
        return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    poly_sets = [
        [rect(0, 0, 10, 10)],
        [rect(0, 0, 100, 10), rect(5, 1, 105, 9)],
        [rect(0, 0, 10, 10), [[0, 0], [10, 0], [12, 20], [0, 10]]],
        [rect(0, 0, 10, 10), rect(0, 20, 10, 30), rect(3, 1, 8, 9)],
        # lb 宽 20 高 10，rt=2 时恰相等（严格 > 不触发）
        [rect(0, 0, 20, 10)],
        # 两个框 x 边界相邻（10 == 10），严格 < 判为重叠并合并
        [rect(0, 0, 10, 10), rect(10, 0, 100, 10)],
        [],
    ]
    n = 0
    for ps in poly_sets:
        for rt in (2, 4, 10):
            for ot in (0.6, 0.8):
                got = solve(mode="det_box_merge", polys=ps,
                            ratio_threshold=rt, overlap_threshold=ot)
                exp = ora_det(ps, rt, ot)
                assert got == exp, ("det ps=%r rt=%r ot=%r\n got=%r\n exp=%r"
                                    % (ps, rt, ot, got, exp))
                n += 1
    # 手算锚点 1：宽高比 105 > 10*4 触发同行横向合并
    assert solve(mode="det_box_merge", polys=[
        rect(0, 0, 100, 10), rect(5, 1, 105, 9)],
        ratio_threshold=4, overlap_threshold=0.6) == {
        "angle_flags": [False, False],
        "horizontal_bboxes": [[0, 0, 100, 10], [5, 1, 105, 9]],
        "line_bboxes": [[0, 0, 105, 10]],
        "ratio_triggered": [True],
        "result_bboxes": [[0, 0, 105, 10]]}
    # 手算锚点 2：angle 框隔离且末尾原样追加；宽高比不触发
    assert solve(mode="det_box_merge", polys=[
        rect(0, 0, 10, 10), [[0, 0], [10, 0], [12, 20], [0, 10]]],
        ratio_threshold=4, overlap_threshold=0.6) == {
        "angle_flags": [False, True],
        "horizontal_bboxes": [[0, 0, 10, 10]],
        "line_bboxes": [[0, 0, 10, 10]],
        "ratio_triggered": [False],
        "result_bboxes": [[0, 0, 10, 10], [0, 0, 10, 20]]}
    # 手算锚点 3：不传 ratio_threshold 时默认 4 生效（10 <= 4*10 不触发）
    assert solve(mode="det_box_merge", polys=[rect(0, 0, 10, 10)])[
        "ratio_triggered"] == [False]
    # 手算锚点 4：不传 overlap_threshold 时默认 0.6 生效（0.7 重叠判同行）
    assert solve(mode="det_box_merge", polys=[
        rect(0, 0, 10, 10), rect(0, 3, 10, 13)])["line_bboxes"] == [
        [0, 0, 10, 13]]
    # 手算锚点 5：x 边界相邻按严格 < 判为重叠并合并
    assert solve(mode="det_box_merge", polys=[
        rect(0, 0, 10, 10), rect(10, 0, 100, 10)],
        ratio_threshold=4, overlap_threshold=0.6)["result_bboxes"] == [
        [0, 0, 100, 10]]
    # 手算锚点 6：宽高比恰等于阈值（20 == 10*2）严格 > 不触发
    assert solve(mode="det_box_merge", polys=[rect(0, 0, 20, 10)],
                 ratio_threshold=2, overlap_threshold=0.6)[
        "ratio_triggered"] == [False]
    print("det_box_merge: %d grid + 6 anchors OK" % n)


def check(solve):
    check_overlap_gate(solve)
    check_interval_ledger(solve)
    check_line_grouping(solve)
    check_det_box_merge(solve)
    print("all checks passed")
