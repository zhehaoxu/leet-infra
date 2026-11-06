"""MinerU 检测框区间与行聚合账（order 836）。

回源：opendatalab/MinerU (master)
  mineru/model/ocr/geometry.py                  重叠门限 / 区间合并与裁除 / angle / 宽高比合并
  mineru/backend/analysis/pdf/text/lines.py     span -> line 分组（竖排判定 / 特殊类型断行 / 行内排序）

判题约定：单入口 solve(mode, **kw)，4 个 mode：
  overlap_gate      y/x 轴重叠比例门限（严格 >，退化框保护）
  interval_ledger   闭区间排序合并 + 遮罩 ±1 离散裁除
  line_grouping     INTERLINE 降级 + 竖排判定 + 特殊类型断行 + 行内排序 + 外接矩形
  det_box_merge     angle 框隔离 + 宽高比 > 4 触发同行横向合并

不进判题域（避免与既有题重复 / 需真模型）：layout/VLM/OCR 真检测、numpy 四点数组、
sorted_boxes 阅读顺序（撞 ocr-layout-column-sort）、title_leveling 的 LLM 分级、
_natural_language_spans 的 LaTeX 定界符切分。bbox/poly 一律数值入参，纯 Python。
"""

OVERLAP_DEFAULT = 0.8
LINE_OVERLAP = 0.6
VERTICAL_RATIO = 2
VERTICAL_DOMINANCE = 0.8
LINE_WIDTH_TO_HEIGHT = 4
SPECIAL_TYPES = {"interline_equation", "image", "table"}


# ---------------------------------------------------------------- 基础
def _overlap_gate(a, b, axis, threshold):
    """geometry.py::_is_overlaps_{y,x}_exceeds_threshold 的可观测版。"""
    if axis == "y":
        overlap = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        e1, e2 = a[3] - a[1], b[3] - b[1]
    else:
        overlap = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        e1, e2 = a[2] - a[0], b[2] - b[0]
    min_extent = min(e1, e2)
    if min_extent <= 0:
        return {"overlap": overlap, "min_extent": min_extent,
                "ratio": None, "exceeds": False, "zero_extent": True}
    ratio = overlap / min_extent
    return {"overlap": overlap, "min_extent": min_extent,
            "ratio": ratio, "exceeds": ratio > threshold,
            "zero_extent": False}


def _merge_intervals(intervals):
    """geometry.py::merge_intervals（用输入副本，不判原地副作用）。"""
    ordered = sorted(intervals, key=lambda i: i[0])
    merged = []
    for it in ordered:
        if not merged or merged[-1][1] < it[0]:
            merged.append([it[0], it[1]])
        else:
            merged[-1][1] = max(merged[-1][1], it[1])
    return merged


def _remove_intervals(original, masks):
    """geometry.py::remove_intervals；±1 为离散整数点语义。"""
    result = []
    start, end = original
    cur = start
    for mask_start, mask_end in _merge_intervals(masks):
        if mask_start > end or mask_end < start:
            continue
        if cur < mask_start:
            result.append([cur, mask_start - 1])
        cur = max(mask_end + 1, cur)
    if cur <= end:
        result.append([cur, end])
    return result


def _merge_overlapping_spans(bboxes):
    """geometry.py::merge_overlapping_spans（按 x0 排序的同行合并）。"""
    ordered = sorted(bboxes, key=lambda b: b[0])
    merged = []
    for b in ordered:
        if not merged or merged[-1][2] < b[0]:
            merged.append([b[0], b[1], b[2], b[3]])
        else:
            prev = merged.pop()
            merged.append([min(prev[0], b[0]), min(prev[1], b[1]),
                           max(prev[2], b[2]), max(prev[3], b[3])])
    return merged


def _is_angle(poly):
    """geometry.py::calculate_is_angle。"""
    p1, p2, p3, p4 = poly
    height = ((p4[1] - p1[1]) + (p3[1] - p2[1])) / 2
    return not (0.8 * height <= (p3[1] - p1[1]) <= 1.2 * height)


def _points_to_bbox(poly):
    """geometry.py::points_to_bbox（p1 左上 / p2 右上 / p3 右下 / p4 左下）。"""
    return [poly[0][0], poly[0][1], poly[1][0], poly[2][1]]


def _line_bbox(line):
    return [min(s[0] for s in line), min(s[1] for s in line),
            max(s[2] for s in line), max(s[3] for s in line)]


# ---------------------------------------------------------------- modes
def _interval_ledger(intervals, masks, original):
    merged_intervals = _merge_intervals(intervals)
    merged_masks = _merge_intervals(masks)
    if original is None:
        remaining = []
        covered = 0
        remaining_points = 0
    else:
        remaining = _remove_intervals(original, masks)
        total = max(0, original[1] - original[0] + 1)
        remaining_points = sum(max(0, e - s + 1) for s, e in remaining)
        covered = total - remaining_points
    return {"merged_intervals": merged_intervals,
            "merged_masks": merged_masks,
            "remaining": remaining,
            "n_segments": len(remaining),
            "covered_points": covered,
            "remaining_points": remaining_points}


def _line_grouping(spans, threshold):
    work = [{"bbox": list(s["bbox"]), "type": s["type"]} for s in spans]
    downgraded = 0
    for s in work:
        if s["type"] == "interline_equation":
            s["type"] = "inline_equation"
            downgraded += 1

    valid = 0
    vertical = 0
    for s in work:
        b = s["bbox"]
        if not b or len(b) < 4:
            continue
        w, h = b[2] - b[0], b[3] - b[1]
        if w <= 0 or h <= 0:
            continue
        valid += 1
        if h / w > VERTICAL_RATIO:
            vertical += 1
    is_vert = False if valid == 0 else vertical / valid > VERTICAL_DOMINANCE

    if is_vert:
        ordered = sorted(work, key=lambda s: s["bbox"][2], reverse=True)
        axis = "x"
        inner_key = lambda s: s["bbox"][1]
        orientation = "vertical"
    else:
        ordered = sorted(work, key=lambda s: s["bbox"][1])
        axis = "y"
        inner_key = lambda s: s["bbox"][0]
        orientation = "horizontal"

    lines = []
    if ordered:
        current = [ordered[0]]
        for s in ordered[1:]:
            if (s["type"] in SPECIAL_TYPES
                    or any(it["type"] in SPECIAL_TYPES for it in current)):
                lines.append(current)
                current = [s]
            elif _overlap_gate(s["bbox"], current[-1]["bbox"],
                               axis, threshold)["exceeds"]:
                current.append(s)
            else:
                lines.append(current)
                current = [s]
        lines.append(current)

    for line in lines:
        line.sort(key=inner_key)
    out_lines = [[list(s["bbox"]) for s in line] for line in lines]
    out_bboxes = [_line_bbox([s["bbox"] for s in line]) for line in lines]
    return {"vertical": is_vert, "valid_span_count": valid,
            "vertical_span_count": vertical, "downgraded": downgraded,
            "lines": out_lines, "line_bboxes": out_bboxes,
            "orientation": orientation}


def _det_box_merge(polys, ratio_threshold, overlap_threshold):
    angle_flags = []
    horizontal = []
    angle_polys = []
    for poly in polys:
        if _is_angle(poly):
            angle_flags.append(True)
            angle_polys.append(poly)
        else:
            angle_flags.append(False)
            horizontal.append(_points_to_bbox(poly))

    ordered = sorted(horizontal, key=lambda b: b[1])
    lines = []
    if ordered:
        current = [ordered[0]]
        for b in ordered[1:]:
            if _overlap_gate(b, current[-1], "y", overlap_threshold)["exceeds"]:
                current.append(b)
            else:
                lines.append(current)
                current = [b]
        lines.append(current)

    line_bboxes = []
    ratio_triggered = []
    result = []
    for line in lines:
        lb = _line_bbox(line)
        line_bboxes.append(lb)
        trig = (lb[2] - lb[0]) > (lb[3] - lb[1]) * ratio_threshold
        ratio_triggered.append(trig)
        if trig:
            result.extend(_merge_overlapping_spans(line))
        else:
            result.extend([list(b) for b in line])
    result.extend(_points_to_bbox(p) for p in angle_polys)
    return {"angle_flags": angle_flags,
            "horizontal_bboxes": [list(b) for b in horizontal],
            "line_bboxes": line_bboxes,
            "ratio_triggered": ratio_triggered,
            "result_bboxes": [list(b) for b in result]}


def solve(mode: str, **kw):
    if mode == "overlap_gate":
        return _overlap_gate(list(kw["bbox1"]), list(kw["bbox2"]),
                             kw.get("axis", "y"),
                             kw.get("threshold", OVERLAP_DEFAULT))
    if mode == "interval_ledger":
        return _interval_ledger(list(kw["intervals"]),
                                list(kw.get("masks", [])),
                                kw.get("original"))
    if mode == "line_grouping":
        return _line_grouping(list(kw["spans"]),
                              kw.get("threshold", LINE_OVERLAP))
    if mode == "det_box_merge":
        return _det_box_merge(list(kw["polys"]),
                              kw.get("ratio_threshold", LINE_WIDTH_TO_HEIGHT),
                              kw.get("overlap_threshold", LINE_OVERLAP))
    raise ValueError("unknown mode: %s" % mode)
