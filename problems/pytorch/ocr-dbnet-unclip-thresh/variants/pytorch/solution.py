import math


def _shoelace(box):
    """有向面积（shapely Polygon.area 取绝对值；顶点序无关）。"""
    s = 0.0
    n = len(box)
    for i in range(n):
        x1, y1 = box[i]
        x2, y2 = box[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return s / 2.0


def _perimeter(box):
    """闭合周长（含末点->首点；shapely Polygon.length 语义）。"""
    total = 0.0
    n = len(box)
    for i in range(n):
        x1, y1 = box[i]
        x2, y2 = box[(i + 1) % n]
        total += math.hypot(x2 - x1, y2 - y1)
    return total


def solve(pred, box, thresh=0.3, box_thresh=0.7, unclip_ratio=2.0,
          min_size=3, dest_width=None, dest_height=None):
    """PaddleOCR DB 后处理单框账（ppocr/postprocess/db_postprocess.py
    DBPostProcess.boxes_from_bitmap 的 quad 流程，输入 contour 已
    get_mini_boxes 化为轴对齐矩形四点；旋转矩形/多边形光栅化不入题，
    见 notes）。

    官方口径（逐条钉死，语义经 shapely 2.2 + pyclipper 1.4 + cv2 5.0
    实测）：

      二值化   segmentation = pred > thresh（严格 >，逐点）
      sside 门 get_mini_boxes 的 sside = min(span_x, span_y)
               （minAreaRect size）；sside < min_size -> skip，score
               /distance/expanded 均不算（账面 None）
      score    box_score_fast：bbox 四向 xmin=clip(floor(min x),0,W-1)
               / xmax=clip(ceil(max x),0,W-1)（y 同 H）；mask 用
               fillPoly——坐标 astype(int32) **向零截断**（非 floor）
               后闭多边形（轴对齐矩形 = 闭矩形 [tx_lo,tx_hi]x[ty_lo,
               ty_hi] 含右/下边界），与 bbox 区域求交；score =
               sum(prob*mask)/mask_sum，mask 全 0 时 cv2.mean = 0.0
      score 门 box_thresh > score -> skip（严格 >）
      distance shapely：|shoelace| * unclip_ratio / 闭合周长
               （Polygon.area 顶点序无关取绝对值）
      expanded pyclipper JT_ROUND：**输入坐标向零截断**，输出坐标
               round-half-away（floor(v+0.5)，与 numpy 银行家不同：
               50.5->51）；轴对齐矩形外扩 = x_lo = r(trunc(min x) -
               distance)，x_hi = r(trunc(max x) + distance)（y 同）
      sside 门 unclip 后 sside < min_size + 2 -> skip
      dest 缩放 box[:,0] = clip(round(x/W*dest_width), 0, dest_width)
               （numpy round = **银行家舍入**：6.5->6、19.5->20；
               W 为 prob map 宽度）

    参数
    ----
    pred : (H, W) 概率图（list[list[float]]）
    box  : 轴对齐矩形四点 [[x,y]*4]，顶点序任意（面积取绝对值）；
           pyclipper 输入坐标的截断语义要求**整数或截断后相同的坐
           标**——浮点非整坐标的外扩有 JT_ROUND 圆弧离散 ±1 误差，
           不入题（notes）
    dest_width/dest_height : 给定则做缩放账（scaled 键），否则 None

    返回 dict（键固定）：
      bitmap / pos_count / sside_in /
      score / bbox（gate1 失败为 None）/ distance / expanded /
      sside_out / gate（"min_size_in"|"box_thresh"|"min_size_out"|None）
      / passed / scaled（None 或缩放后四点）

    红线：不得 import cv2 / shapely / pyclipper；纯标量账。
    """
    H = len(pred)
    W = len(pred[0]) if H else 0
    if len(box) != 4 or any(len(p) != 2 for p in box):
        raise ValueError("box must be 4 points of [x, y]")
    if any(v < 0 or v > 1 for row in pred for v in row):
        raise ValueError("pred must be probabilities in [0, 1]")

    bitmap = [[p > thresh for p in row] for row in pred]
    pos_count = sum(sum(1 for p in row if p) for row in bitmap)

    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    x_lo, x_hi = min(xs), max(xs)
    y_lo, y_hi = min(ys), max(ys)
    sside_in = min(x_hi - x_lo, y_hi - y_lo)

    out = {"bitmap": bitmap, "pos_count": pos_count,
           "sside_in": sside_in, "score": None, "bbox": None,
           "distance": None, "expanded": None, "sside_out": None,
           "gate": None, "passed": False, "scaled": None}

    # --- sside 门（get_mini_boxes，先于 score） ---
    if sside_in < min_size:
        out["gate"] = "min_size_in"
        return out

    # --- box_score_fast：bbox floor/ceil/clip + fillPoly 截断闭矩形 ---
    xmin = min(max(math.floor(x_lo), 0), W - 1)
    xmax = min(max(math.ceil(x_hi), 0), W - 1)
    ymin = min(max(math.floor(y_lo), 0), H - 1)
    ymax = min(max(math.ceil(y_hi), 0), H - 1)
    out["bbox"] = [xmin, xmax, ymin, ymax]
    # fillPoly 坐标 astype(int32) 向零截断
    tx_lo, tx_hi = math.trunc(x_lo), math.trunc(x_hi)
    ty_lo, ty_hi = math.trunc(y_lo), math.trunc(y_hi)
    mx_lo, mx_hi = max(xmin, tx_lo), min(xmax, tx_hi)
    my_lo, my_hi = max(ymin, ty_lo), min(ymax, ty_hi)
    total, cnt = 0.0, 0
    for yy in range(my_lo, my_hi + 1):
        for xx in range(mx_lo, mx_hi + 1):
            total += pred[yy][xx]
            cnt += 1
    score = total / cnt if cnt else 0.0
    out["score"] = score

    if box_thresh > score:
        out["gate"] = "box_thresh"
        return out

    # --- unclip：distance 公式 + pyclipper 整数坐标语义 ---
    area = _shoelace(box)
    perim = _perimeter(box)
    if perim == 0:
        raise ValueError("degenerate box (zero perimeter)")
    distance = abs(area) * unclip_ratio / perim

    def pc_round(v):        # pyclipper 输出：round half away from zero
        return math.floor(v + 0.5)

    bx_lo = pc_round(math.trunc(x_lo) - distance)
    bx_hi = pc_round(math.trunc(x_hi) + distance)
    by_lo = pc_round(math.trunc(y_lo) - distance)
    by_hi = pc_round(math.trunc(y_hi) + distance)
    expanded = [[bx_lo, by_lo], [bx_hi, by_lo],
                [bx_hi, by_hi], [bx_lo, by_hi]]
    sside_out = min(bx_hi - bx_lo, by_hi - by_lo)
    out["distance"] = distance
    out["expanded"] = expanded
    out["sside_out"] = sside_out

    if sside_out < min_size + 2:
        out["gate"] = "min_size_out"
        return out

    out["passed"] = True
    if dest_width is not None and dest_height is not None:
        scaled = []
        for px, py in expanded:
            sx = min(max(round(px / W * dest_width), 0), dest_width)
            sy = min(max(round(py / H * dest_height), 0), dest_height)
            scaled.append([sx, sy])
        out["scaled"] = scaled
    return out
