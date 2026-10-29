import torch


def solve(W0, A, B, lam, w_norm):
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A 与 B 都必须是二维")
    r, in_features = A.shape
    out_features, r_b = B.shape
    if r_b != r:
        raise ValueError("B 的第二维必须等于 A 的第一维（同一个 rank）")
    if W0.ndim != 2 or W0.shape != (out_features, in_features):
        raise ValueError("W0 必须是 (out_features, in_features)")
    if w_norm.ndim != 1 or w_norm.numel() != in_features:
        raise ValueError("w_norm 必须是一维且长度等于 in_features")
    if r < 1:
        raise ValueError("rank 必须 >= 1")
    if lam < 0:
        raise ValueError("lam 必须 >= 0")

    af = A.to(torch.float64)
    bf = B.to(torch.float64)
    w0f = W0.to(torch.float64)
    wn = w_norm.to(torch.float64)

    an = af.norm(dim=1).clamp(min=1e-4)
    bn = bf.norm(dim=0).clamp(min=1e-4)
    scaling = (float(lam) / r) / (an * bn)
    delta = ((bf * scaling.unsqueeze(0)) @ af) * wn.unsqueeze(0)
    merged = w0f + delta
    return scaling.to(torch.float32), delta.to(torch.float32), merged.to(torch.float32)
