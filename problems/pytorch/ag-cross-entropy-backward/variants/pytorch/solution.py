"""参考答案：p − (1−ε)·onehot − ε/C，逐行掩掉 ignore，再按有效行数归一。

要点：softmax 与 one-hot 的差就是交叉熵对 logits 的梯度；label smoothing 相当于把
one-hot 目标换成 (1−ε)·onehot + ε/C 的软目标，于是梯度只需把硬目标替换成软目标，
p 那一项不变。ignore_index 的口径最容易错：整行置 0 还不够，reduction='mean' 的
分母必须同时剔除被忽略的行，否则有效行的梯度会被稀释。
"""

import torch


def solve(logits, target, label_smoothing=0.0, ignore_index=-100, reduction='mean'):
    if logits.dim() != 2:
        raise ValueError("logits 必须是 (N, C) 二维张量")
    if target.dim() != 1 or target.shape[0] != logits.shape[0]:
        raise ValueError("target 必须是长度为 N 的一维张量")
    if not 0.0 <= float(label_smoothing) < 1.0:
        raise ValueError("label_smoothing 必须在 [0, 1) 内")
    if reduction not in ('mean', 'sum'):
        raise ValueError("reduction 只支持 'mean' 或 'sum'")

    n, c = logits.shape
    eps = float(label_smoothing)
    x = logits.double()
    p = torch.softmax(x, dim=1)

    valid = target != ignore_index
    idx = torch.where(valid, target.clamp(min=0), torch.zeros_like(target))
    onehot = torch.zeros_like(p)
    onehot.scatter_(1, idx.unsqueeze(1), 1.0)
    onehot = onehot * valid.unsqueeze(1).to(p.dtype)

    grad = p - (1.0 - eps) * onehot - eps / c
    grad = torch.where(valid.unsqueeze(1), grad, torch.zeros_like(grad))

    if reduction == 'mean':
        cnt = int(valid.sum())
        if cnt > 0:
            grad = grad / cnt
    return grad.to(logits.dtype)
