import torch


def solve(w, sens, k, iters):
    """SqueezeLLM 敏感度加权 1D Lloyd（float64 内部、float32 输出）。"""
    n = int(w.numel())
    k = int(k)
    if not (1 <= k <= n):
        raise ValueError("k 必须在 [1, N] 内")
    if int(iters) < 0:
        raise ValueError("iters 必须非负")
    w64 = w.to(torch.float64).reshape(-1)
    s64 = sens.to(torch.float64).reshape(-1)
    if bool((s64 < 0).any()) or float(s64.sum()) <= 0:
        raise ValueError("sens 必须非负且总和为正")

    order = torch.argsort(w64)
    ws = w64[order]
    idx = [min(int((j + 0.5) * n / k), n - 1) for j in range(k)]
    cent = ws[torch.tensor(idx, dtype=torch.long)].clone()

    for _ in range(int(iters)):
        assign = (w64[:, None] - cent[None, :]).abs().argmin(dim=1)
        new = cent.clone()
        for j in range(k):
            m = assign == j
            if bool(m.any()):
                new[j] = (w64[m] * s64[m]).sum() / s64[m].sum()
        cent = new.sort().values

    assign = (w64[:, None] - cent[None, :]).abs().argmin(dim=1)
    return (cent.to(torch.float32), assign.to(torch.int64),
            cent[assign].to(torch.float32))
