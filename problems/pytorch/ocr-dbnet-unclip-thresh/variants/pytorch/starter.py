import math


def solve(pred, box, thresh=0.3, box_thresh=0.7, unclip_ratio=2.0,
          min_size=3, dest_width=None, dest_height=None):
    """PaddleOCR DB 后处理单框账（ppocr/postprocess/db_postprocess.py
    DBPostProcess.boxes_from_bitmap 的 quad 流程，contour 已
    get_mini_boxes 化为轴对齐矩形四点）。

    官方口径要点（shapely/pyclipper/cv2 语义实测钉死）：
      - 二值化：segmentation = pred > thresh（严格 >，逐点）
      - sside 门先行：sside = min(span_x, span_y)，sside < min_size
        -> score/distance/expanded 均不算（账面 None）
      - box_score_fast：bbox 四向 floor/ceil + clip [0, W-1]/[0, H-1]；
        mask 用 fillPoly——坐标 astype(int32) **向零截断**后闭矩形
        （含右/下边界）与 bbox 区域求交；score = 区域均值，mask 全 0
        时 cv2.mean 语义 = 0.0
      - score 门：box_thresh > score -> skip（严格 >）
      - distance = |shoelace| * unclip_ratio / 闭合周长（shapely
        Polygon.area 顶点序无关取绝对值）
      - pyclipper JT_ROUND：输入坐标向零截断、输出坐标 round-half-away
        （floor(v+0.5)，与 numpy 银行家不同：50.5->51）；轴对齐矩形
        外扩 = 每边 ±distance 后取整
      - unclip 后 sside < min_size + 2 -> skip
      - dest 缩放：clip(round(x/W*dest_width), 0, dest_width)（numpy
        round = **银行家舍入**：6.5->6、19.5->20）

    返回 dict（键固定）：bitmap / pos_count / sside_in / score / bbox /
    distance / expanded / sside_out / gate（"min_size_in"|"box_thresh"|
    "min_size_out"|None）/ passed / scaled。
    红线：不得 import cv2 / shapely / pyclipper；纯标量账。
    """
    raise NotImplementedError
