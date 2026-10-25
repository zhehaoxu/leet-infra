"""参考答案：2:4 半结构化剪枝（每 4 元素组内 topk(2) by |w|）。"""

import torch


def solve(weight):
    orig = weight.shape
    g = weight.reshape(-1, 4)
    _, top = g.abs().topk(2, dim=1)
    mask = torch.zeros_like(g)
    mask.scatter_(1, top, 1.0)
    return (g * mask).reshape(orig)
