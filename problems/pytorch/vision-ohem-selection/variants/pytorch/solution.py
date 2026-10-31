"""OHEM 难例选择掩码账（solution）。

转写 mmseg OHEMPixelSampler（1604.03540 OHEM 的像素级主流口径）。
"""

import math


def solve(losses, probs, thresh, min_kept, batch_size):
    if not isinstance(min_kept, int) or min_kept < 1:
        raise ValueError("min_kept must be int >= 1")
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be int >= 1")
    if thresh is not None:
        if not isinstance(thresh, (int, float)) or not math.isfinite(thresh) \
                or thresh < 0 or thresh > 1:
            raise ValueError("thresh must be a float in [0, 1]")
    batch_kept = min_kept * batch_size

    if thresh is not None:
        # threshold 分支：输入正确类概率
        if losses is not None:
            raise ValueError("loss branch must pass probs=None")
        if not isinstance(probs, list) or not all(
            isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1
            for v in probs
        ):
            raise ValueError("probs must be floats in [0, 1]")
        n = len(probs)
        if n > 0:
            sorted_prob = sorted(probs)  # 升序
            min_threshold = sorted_prob[min(batch_kept, n - 1)]
        else:
            min_threshold = 0.0
        threshold = max(min_threshold, thresh)
        mask = [1 if v < threshold else 0 for v in probs]
        return {"mask": mask, "threshold": threshold}

    # loss 分支
    if probs is not None:
        raise ValueError("loss branch must pass probs=None")
    if not isinstance(losses, list) or not all(
        isinstance(v, (int, float)) and math.isfinite(v) and v >= 0
        for v in losses
    ):
        raise ValueError("losses must be non-negative finite floats")
    # 官方：sort(descending=True) 的稳定排序，取前 batch_kept
    order = sorted(range(len(losses)), key=lambda i: (-losses[i], i))
    mask = [0] * len(losses)
    for i in order[:batch_kept]:
        mask[i] = 1
    return {"mask": mask, "threshold": None}
