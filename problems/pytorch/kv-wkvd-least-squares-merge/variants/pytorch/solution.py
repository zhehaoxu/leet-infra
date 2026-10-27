import torch


def solve(V_keep, V_evict):
    """最小二乘合并 + 伪逆（float64 求解、float32 输出）。"""
    if V_keep.ndim != 2 or V_evict.ndim != 2:
        raise ValueError("输入必须是二维")
    if V_keep.shape[0] < 1 or V_evict.shape[0] < 1:
        raise ValueError("K 与 E 都必须 >= 1")
    if V_keep.shape[1] != V_evict.shape[1]:
        raise ValueError("特征维 d 必须一致")

    keep = V_keep.to(torch.float64)
    evict = V_evict.to(torch.float64)
    basis = keep.T                                   # (d, K)
    target = evict.T                                 # (d, E)
    coef = torch.linalg.pinv(basis) @ target         # (K, E)
    merged = keep + coef @ evict                     # (K, d)
    residual = torch.linalg.matrix_norm(basis @ coef - target)
    return (merged.to(torch.float32), coef.to(torch.float32),
            residual.to(torch.float32).reshape(()))
