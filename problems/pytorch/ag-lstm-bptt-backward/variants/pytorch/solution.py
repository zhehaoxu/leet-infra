"""参考答案：先把整条时间线前向存下来，再从 T−1 往回走一段标准 BPTT。

两个最容易错的地方：
1) dh/dc 的双路累加：dCt = dc_{t+1} + dHt·(1 − tanh(c_t)²)，而 dHt 本身是「上游给的
   d h_{t+1}」与本步 c/h 递推产生的分量之和，两段都不能漏；只走一条路会在 T>1 时全错。
2) 权重梯度是跨时间的求和：dW_ih = Σ_t dgates_tᵀ @ x_t。逐时间步各自返回是错的口径。
门顺序按 PyTorch 权重排布 (i, f, g, o) 切 4 段，σ′ 用 y(1−y)、tanh′ 用 1 − y²。
"""

import torch


def solve(xs, h0, c0, W_ih, W_hh, b_ih, b_hh, dhT, dcT):
    if xs.dim() != 3:
        raise ValueError("xs 必须是 (T, N, E) 三维张量")
    if h0.dim() != 2 or c0.dim() != 2:
        raise ValueError("h0/c0 必须是 (N, Hid) 二维张量")
    if W_ih.dim() != 2 or W_hh.dim() != 2:
        raise ValueError("W_ih/W_hh 必须是二维")
    if b_ih.dim() != 1 or b_hh.dim() != 1:
        raise ValueError("b_ih/b_hh 必须是一维")
    T, N, E = xs.shape
    Hid = h0.shape[1]
    if c0.shape != (N, Hid):
        raise ValueError("c0 必须与 h0 同形")
    if tuple(W_ih.shape) != (4 * Hid, E):
        raise ValueError("W_ih 必须是 (4*Hid, E)")
    if tuple(W_hh.shape) != (4 * Hid, Hid):
        raise ValueError("W_hh 必须是 (4*Hid, Hid)")
    if tuple(b_ih.shape) != (4 * Hid,) or tuple(b_hh.shape) != (4 * Hid,):
        raise ValueError("b_ih/b_hh 必须是 (4*Hid,)")
    if tuple(dhT.shape) != (N, Hid) or tuple(dcT.shape) != (N, Hid):
        raise ValueError("dhT/dcT 必须是 (N, Hid)")

    x64 = xs.double()
    h_prev0 = h0.double()
    c_prev0 = c0.double()
    w_ih = W_ih.double()
    w_hh = W_hh.double()
    bias = b_ih.double() + b_hh.double()
    dh_next = dhT.double()
    dc_next = dcT.double()

    # ---- 前向：把每步的 i/g/f/o、c_{t−1}、c_t、tanh(c_t) 全部存下来 ----
    cache = []
    h_prev, c_prev = h_prev0, c_prev0
    for t in range(T):
        gates = x64[t] @ w_ih.T + h_prev @ w_hh.T + bias
        i = torch.sigmoid(gates[:, 0:Hid])            # PyTorch 门顺序：i, f, g(cell), o
        f = torch.sigmoid(gates[:, Hid:2 * Hid])
        g = torch.tanh(gates[:, 2 * Hid:3 * Hid])
        o = torch.sigmoid(gates[:, 3 * Hid:4 * Hid])
        c = f * c_prev + i * g
        tc = torch.tanh(c)
        h = o * tc
        cache.append((x64[t], h_prev, c_prev, i, g, f, o, c, tc))
        h_prev, c_prev = h, c

    # ---- 反向：时间轴倒着走，累加权重/偏置梯度 ----
    dW_ih = torch.zeros(4 * Hid, E, dtype=torch.float64)
    dW_hh = torch.zeros(4 * Hid, Hid, dtype=torch.float64)
    db = torch.zeros(4 * Hid, dtype=torch.float64)
    dxs = torch.zeros(T, N, E, dtype=torch.float64)
    for t in range(T - 1, -1, -1):
        x_t, h_p, c_p, i, g, f, o, c, tc = cache[t]
        dh = dh_next
        # h_t = o ⊙ tanh(c_t) ⇒ 走 h 这条路回传到 c 时必须再乘一次 o
        dct = dc_next + dh * o * (1.0 - tc * tc)
        di = dct * g
        dg = dct * i
        df = dct * c_p
        do = dh * tc                      # h = o⊙tanh(c) ⇒ 这里乘 tanh(c)，不是 c
        dgates = torch.cat([
            di * i * (1.0 - i),             # σ′(z) = y(1−y)
            df * f * (1.0 - f),
            dg * (1.0 - g * g),             # tanh′(z) = 1 − y²
            do * o * (1.0 - o),
        ], dim=1)
        dxs[t] = dgates @ w_ih
        dh_next = dgates @ w_hh
        dc_next = dct * f
        dW_ih += dgates.T @ x_t
        dW_hh += dgates.T @ h_p
        db += dgates.sum(dim=0)

    dh0, dc0 = dh_next, dc_next
    # b_ih 与 b_hh 都完整加进 gates ⇒ 两者的梯度都是同一个 Σ_t dgates（4·Hid 全长）
    db_full = db.to(b_ih.dtype)
    return (dxs.to(xs.dtype), dW_ih.to(W_ih.dtype), dW_hh.to(W_hh.dtype),
            db_full.clone(), db_full.clone().to(b_hh.dtype),
            dh0.to(h0.dtype), dc0.to(c0.dtype))
