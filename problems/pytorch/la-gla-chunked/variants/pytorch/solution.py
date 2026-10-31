import torch


def solve(q, k, v, g, chunk_size, h0=None, scale=None):
    for name, t, shape in (("q", q, 2), ("k", k, 2), ("v", v, 2), ("g", g, 2)):
        if not torch.is_tensor(t) or t.dim() != shape:
            raise ValueError(f"{name} must be a 2-D tensor")
    if not torch.is_tensor(chunk_size) and not isinstance(chunk_size, int):
        raise ValueError("chunk_size must be a positive int")
    if isinstance(chunk_size, bool) or int(chunk_size) != chunk_size or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive int")
    T, K = q.shape
    if k.shape != (T, K) or g.shape != (T, K):
        raise ValueError("q/k/g shape mismatch")
    V = v.shape[1]
    if v.shape[0] != T:
        raise ValueError("v row count must match T")
    if V == 0 or K == 0 or T == 0:
        raise ValueError("empty input")
    if scale is None:
        scale = K ** -0.5
    else:
        scale = float(scale)
        if not (scale > 0) or scale != scale or scale in (float("inf"),):
            raise ValueError("scale must be a positive finite number")
    if h0 is not None:
        if not torch.is_tensor(h0) or h0.shape != (K, V):
            raise ValueError("h0 must be [K, V]")

    qf = q.double()
    kf = k.double()
    vf = v.double()
    gf = g.double()

    h = torch.zeros(K, V, dtype=torch.float64) if h0 is None else h0.double().clone()
    L = int(chunk_size)
    o = torch.zeros(T, V, dtype=torch.float64)
    chunk_states = []
    for c0 in range(0, T, L):
        c1 = min(c0 + L, T)
        gc = torch.cumsum(gf[c0:c1], dim=0)      # S_t，段内含入 cumsum
        qc = qf[c0:c1] * scale
        kc = kf[c0:c1]
        vc = vf[c0:c1]
        # 段间贡献：读段前状态，逐 key 通道衰减 exp(S_t)
        qg = qc * torch.exp(gc)
        o[c0:c1] += qg @ h
        # 段内贡献：A[t,j] = Σ_i q_t,i·k_j,i·exp(S_t,i - S_j,i)，下三角含对角
        A = (qc * torch.exp(gc)) @ (kc * torch.exp(-gc)).T
        l = c1 - c0
        mask = torch.tril(torch.ones(l, l, dtype=torch.bool))
        o[c0:c1] += torch.where(mask, A, torch.zeros_like(A)) @ vc
        # 状态更新：段末衰减（按 key 通道逐行）+ 带剩余衰减的 k 累积
        dec_end = torch.exp(gc[-1] - gc)
        h = torch.exp(gc[-1])[:, None] * h + (kc * dec_end).T @ vc
        chunk_states.append(h.clone())
    return {"o": o, "final_state": h, "chunk_states": chunk_states}
