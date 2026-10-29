import torch


def solve(u, W0, apply_gs, reverse=False):
    u = u.to(torch.float64)
    W0f = W0.to(torch.float64)
    if u.ndim != 2:
        raise ValueError("u 必须是二维 (in_features, r)")
    if W0f.ndim != 2 or W0f.shape[1] != u.shape[0]:
        raise ValueError("W0 必须是 (out_features, in_features) 且 in_features 与 u 的行数一致")
    n, r = u.shape
    if r < 1:
        raise ValueError("r 必须 >= 1")
    norms = u.norm(dim=0)
    if bool((norms == 0).any()):
        raise ValueError("u 的每一列都必须非零")
    if apply_gs and r > n:
        raise ValueError("apply_gs=True 时需要 r <= in_features")

    eye_r = torch.eye(r, dtype=torch.float64)
    if apply_gs:
        v = torch.linalg.qr(u, mode="reduced").Q
        t = 2 * eye_r
    else:
        v = u / norms.unsqueeze(0)
        tinv = 0.5 * eye_r + torch.triu(v.T @ v, diagonal=1)
        t = torch.linalg.solve_triangular(tinv, eye_r, upper=True)

    if reverse:
        t = t.T

    H = torch.eye(n, dtype=torch.float64) - (v @ t) @ v.T
    merged = W0f @ H
    return (
        v.to(torch.float32),
        t.to(torch.float32),
        H.to(torch.float32),
        merged.to(torch.float32),
    )
