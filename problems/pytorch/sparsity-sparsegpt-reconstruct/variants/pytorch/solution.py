"""参考答案：SparseGPT 逐列掩码 + OBS 误差补偿（把剪枝误差按 Hinv 行推给后续列）。"""

import torch


def solve(weight, mask, Hinv):
    w = weight.clone()
    r, c = w.shape
    for j in range(c):
        cur = w[:, j].clone()
        q = cur * mask[:, j]
        e = (cur - q) / Hinv[j, j]
        w[:, j] = q
        if j + 1 < c:
            w[:, j + 1 :] -= e.unsqueeze(1) * Hinv[j, j + 1 :].unsqueeze(0)
    return w
