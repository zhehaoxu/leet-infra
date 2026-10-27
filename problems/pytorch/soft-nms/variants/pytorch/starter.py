import torch


def solve(boxes, scores, method="linear", sigma=0.5, iou_thr=0.3, score_thr=0.001):
    """Soft-NMS（Bodla et al., ICCV 2017）置信度连续衰减版非极大值抑制。

    入参：
    - boxes: float 张量，形状 (N,4)，box 口径 [x1, y1, x2, y2]，约定均为合法框
      （x2 > x1 且 y2 > y1，不做校验）。N = 0 允许。
    - scores: float 张量，形状 (N,)，与 boxes 一一对应的置信度。
    - method: str，默认 "linear"，取以下两种之一，其余取值抛 ValueError：
      * "linear"：仅当 IoU > iou_thr（严格大于）时衰减 s ← s·(1 − IoU)，否则不变；
      * "gaussian"：无条件衰减 s ← s·exp(−IoU²/sigma)，不看 iou_thr。
    - sigma: float，默认 0.5，仅 "gaussian" 使用；method="gaussian" 且 sigma <= 0 时
      抛 ValueError。method="linear" 时 sigma 完全忽略。
    - iou_thr: float，默认 0.3，仅 "linear" 使用的硬阈值。
    - score_thr: float，默认 0.001：每轮衰减结束后删除当前分数 < score_thr 的
      剩余框（严格小于才删，等于阈值保留）。

    迭代流程（每轮）：
    1. 在剩余框中取当前分数最高者（并列取原始下标最小者）加入保留序，
       记录它「被选中那一刻」（即已经历此前各轮衰减）的分数；
    2. 对剩余每个框，按它与刚选中框的 IoU 衰减分数（规则见 method）；
    3. 删除衰减后分数 < score_thr 的剩余框；
    4. 重复直到剩余框为空。

    IoU 口径与 iou-compute 一致：交集宽高 clamp_min(0)，inter = iw·ih；
    并集 = areaA + areaB − inter，并集为 0 时 IoU 记 0。

    返回 tuple (keep_indices, keep_scores)：
    - keep_indices: long 张量，形状 (M,)，被选中时的原始下标，按选中顺序排列；
    - keep_scores: float 张量，形状 (M,)，对应框被选中那一刻（衰减后）的分数。
    - N = 0 时返回两个 (0,) 空张量；N >= 1 时首轮最高分框必被保留（M >= 1），
      极端情况下其余框可全被 score_thr 剔光。
    """
    # TODO
    raise NotImplementedError
