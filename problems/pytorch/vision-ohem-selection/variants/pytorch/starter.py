"""OHEM 难例选择掩码账（starter）。

口径钉死（1604.03540 论文算法；转写按 mmseg OHEMPixelSampler 双分支）：

- batch_kept = min_kept * batch_size（本题 batch_size 显式入参）。
- **loss 分支**（thresh=None）：losses 升序稳定排序后取**降序前
  batch_kept** 个下标（即 `sort(descending=True)` 的稳定并列取靠前），
  mask 对应位置 1。
- **threshold 分支**（thresh 给定）：输入为每样本**正确类概率** probs，
  升序排序；若 n > 0，min_threshold = sort_prob[min(batch_kept, n-1)]，
  否则 0.0；threshold = max(min_threshold, thresh)；
  mask = probs < threshold（严格小于，保留难例）。
- 输出 {"mask": [0/1...], "threshold": float 或 None}。
- 校验：losses/probs 长度一致且 >= 0 的有限浮点；min_kept >= 1；
  batch_size >= 1；thresh < 0 非法；probs ∈ [0,1]。
"""

import math


def solve(losses, probs, thresh, min_kept, batch_size):
    """losses/probs 二选一（另一个传 None）；返回 mask 与 threshold。"""
    raise NotImplementedError
