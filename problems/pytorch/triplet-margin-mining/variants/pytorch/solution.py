"""参考答案：batch_hard / semihard 在线难例挖掘 + triplet margin 损失。

口径：成对距离 D_ij = ‖x_i − x_j‖_p（逐元素差先取 |·|^p 求和再开 1/p 次方，
不做 cdist 矩阵乘法近似）；正例取同标签（j≠a）中距离最大、负例取异标签中
距离最小，并列一律取最小下标；semihard 在开区间 (d_pos, d_pos+margin) 内取
最小距离负例，区间空则回退全局最难。swap 的 per-triplet 公式按实测与
F.triplet_margin_loss(swap=True) 对齐：relu(margin + d_ap − min(d_an, d_pn))。
"""

import torch

_NEG_INF = float("-inf")
_POS_INF = float("inf")


def solve(embeddings, labels, margin=1.0, p=2.0, swap=False, strategy="batch_hard"):
    if strategy not in ("batch_hard", "semihard"):
        raise ValueError(f"unknown strategy: {strategy!r}")

    n = embeddings.shape[0]
    zero = torch.zeros((), dtype=embeddings.dtype)
    if n == 0:
        return zero

    diff = embeddings.unsqueeze(1) - embeddings.unsqueeze(0)      # (N,N,D)
    dist = diff.abs().pow(p).sum(-1).pow(1.0 / p)                 # (N,N)
    same = labels.unsqueeze(1) == labels.unsqueeze(0)             # same[i,j]: i,j 同标签
    pos_pool = same & ~torch.eye(n, dtype=torch.bool)             # 同标签且 j≠a
    neg_pool = ~same                                              # 异标签（自动排除自身）

    losses = []
    for a in range(n):
        pm, nm = pos_pool[a], neg_pool[a]
        if not bool(pm.any()) or not bool(nm.any()):
            continue
        row = dist[a]
        # 并列时 argmax/argmin 返回首个出现位置（即最小下标）
        j_pos = int(torch.where(pm, row, torch.full_like(row, _NEG_INF)).argmax())
        d_pos = row[j_pos]
        if strategy == "semihard":
            cand = nm & (row > d_pos) & (row < d_pos + margin)
            if not bool(cand.any()):
                cand = nm
        else:
            cand = nm
        j_neg = int(torch.where(cand, row, torch.full_like(row, _POS_INF)).argmin())
        d_neg = row[j_neg]
        if swap:
            d_neg = torch.minimum(d_neg, dist[j_pos, j_neg])
        losses.append(torch.clamp(margin + d_pos - d_neg, min=0.0))

    if not losses:
        return zero
    return torch.stack(losses).mean()
