import torch


def solve(pred_boxes, pred_scores, pred_cls, pred_img, gt_boxes, gt_cls, gt_img,
          iou_thrs=(0.5,), method="coco"):
    """检测 mAP：按类别做贪心 IoU 匹配，再按三种 AP 口径之一积分。

    入参：
    - pred_boxes: (P,4) float32/float64，xyxy；pred_scores: (P,) 同 dtype 置信度；
      pred_cls / pred_img: (P,) 整型，类别下标与图像 id。
    - gt_boxes: (G,4) 同 dtype；gt_cls / gt_img: (G,) 整型。
    - iou_thrs: 非空的浮点序列，每个元素在 [0,1]，逐阈值各算一次 mAP。
    - method: "all"（全程点积分 / VOC2010+）、"11point"（VOC2007 的 11 点）、
      "coco"（COCO 的 101 点），决定 PR 曲线的插值方式。

    单类别单阈值的匹配（图像 id 与类别都要相同才可能匹配）：
    1) 该类别全部预测按置信度降序，同分按原始下标升序；
    2) 依次取出预测，在「同图像、同类别、尚未被占用」的 GT 中取 IoU 最大者
       （IoU 并列取 GT 下标最小者）；若该 IoU >= iou_thr 则匹配成功（TP，占用该 GT），
       否则为 FP。IoU 用裁剪到正的宽高计算，两框面积都为 0 时记 0。

    AP 口径（precision/recall 由上述 TP/FP 序列累积得到，recall 分母是该类别的 GT 数）：
    - "all"：AP = sum_i (r_i - r_{i-1}) * max_{j>=i} p_j，r_{-1}=0；
    - "11point"：在 r = 0, 0.1, ..., 1.0 上取 p(r) = max_{i: r_i>=r} p_i（无则 0），AP 为 11 点均值；
    - "coco"：同上但在 r = 0, 0.01, ..., 1.0 共 101 点上取均值。
    类别集合取「GT 中出现过的类别」（升序），没有 GT 的类别不参与平均；
    某类别一个预测都没有时 AP 记 0。

    返回二元组：
    - ap: (T, C) float64，T = len(iou_thrs)，C = GT 类别数；
    - mean_ap: (T,) float64，每个阈值下的类别平均。
    T 或 C 为 0 时对应维度为 0（mean_ap 在 C=0 时为 0）。

    非法输入一律 ValueError：形状/dtype 不符、各数组长度不一致、iou_thrs 为空或越界、
    method 未知。
    """
    # TODO
    raise NotImplementedError
