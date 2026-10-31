import torch


_WSC_MAP = {
    "min": ("w2", 1.0),
    "ma": ("w3", 0.25),
    "ma2": ("w3", 0.5),
    "2": ("w2", 0.125),
    "1": ("w3", 0.125),
}


def _wsc_threshold(w_flat, choice):
    base, k = _WSC_MAP[choice]
    w2 = float(w_flat.min())
    w3 = float(w_flat.abs().max())
    return (w2 if base == "w2" else w3) * k


def _quantize_rows(Wc, qmax, qmin):
    row_max = Wc.abs().amax(dim=1, keepdim=True)
    scale = row_max / qmax
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    code = torch.clamp(torch.round(Wc / scale), qmin, qmax)
    return code * scale


def solve(W, Xc, alphas, wsc_choices, wsc_w, n_bits, lwc_steps):
    if W.dim() != 2:
        raise ValueError("W must be a 2-D [out, in] matrix")
    if W.shape[0] != W.shape[1]:
        raise ValueError("only square layers (in_features == out_features) are in scope")
    if Xc.dim() != 2:
        raise ValueError("Xc must be a 2-D [tokens, in] matrix")
    if Xc.shape[1] != W.shape[1]:
        raise ValueError("Xc and W disagree on the input-channel dimension")
    if len(alphas) == 0:
        raise ValueError("alphas must not be empty")
    grid = []
    for a in alphas:
        try:
            value = float(a)
        except (TypeError, ValueError):
            raise ValueError("alphas must be real numbers")
        if not value == value or value < 0.0 or value > 1.0:
            raise ValueError("every alpha must lie in [0, 1]")
        grid.append(value)
    if len(wsc_choices) == 0:
        raise ValueError("wsc_choices must not be empty")
    for c in wsc_choices:
        if not isinstance(c, str) or c not in _WSC_MAP:
            raise ValueError("unknown WSC choice, expected one of min/ma/ma2/2/1")
    bits = int(n_bits)
    if bits != n_bits or bits < 2 or bits > 16:
        raise ValueError("n_bits must be an integer in [2, 16]")
    steps = int(lwc_steps)
    if steps != lwc_steps or steps < 0:
        raise ValueError("lwc_steps must be a non-negative integer")
    w_flat = wsc_w.reshape(-1).to(torch.float64)
    if w_flat.numel() == 0:
        raise ValueError("wsc_w must not be empty")

    Wd = W.to(torch.float64)
    Xd = Xc.to(torch.float64)
    n = int(Wd.shape[1])
    tokens = int(Xd.shape[0])
    qmax = 2 ** (bits - 1) - 1
    qmin = -(2 ** (bits - 1))

    s1 = Wd.abs().amax(dim=1)
    s2 = Xd.abs().amax(dim=0)
    eye = torch.eye(n, dtype=torch.float64)
    ones = torch.ones(n, dtype=torch.float64)

    def candidate(alpha):
        s = alpha * s1 + (1.0 - alpha) * s2
        if bool((s == 0).any()):
            raise ValueError("a transform scale s[i] is exactly 0, its reciprocal is undefined")
        shift = torch.zeros_like(s1)
        live = s1 != 0
        shift[live] = alpha * (s1[live] - s[live]) / s1[live]
        T = torch.diag(s) @ (eye - torch.outer(ones, shift) / n)
        T_inv = torch.diag(1.0 / s) @ (eye + torch.outer(ones, shift) / n)
        return s, shift, T, T_inv

    def loss_of(Wt, Wq):
        diff = (Xd @ Wt.t()) - (Xd @ Wq.t())
        return float((diff * diff).sum() / (tokens * n))

    losses = []
    order = []
    for i, alpha in enumerate(grid):
        s, shift, T, T_inv = candidate(alpha)
        Wt = T @ Wd @ T_inv
        for c in wsc_choices:
            thr = _wsc_threshold(w_flat, c)
            Wq = _quantize_rows(torch.clamp(Wt, -thr, thr), qmax, qmin)
            losses.append(loss_of(Wt, Wq))
            order.append((i, c))

    best_flat = min(range(len(losses)), key=losses.__getitem__)
    best_i, best_choice = order[best_flat]
    alpha = grid[best_i]
    s, shift, T, T_inv = candidate(alpha)
    Wt = T @ Wd @ T_inv
    thr = _wsc_threshold(w_flat, best_choice)
    clipped = torch.clamp(Wt, -thr, thr)
    norm_comp = float(((T @ T_inv - eye).abs()).max())

    w_abs = w_flat.abs()
    lwc_alpha = torch.full_like(w_abs, 0.01)
    lwc_bound = w_abs * 2.0 * torch.sigmoid(lwc_alpha)

    return {
        "best_alpha_index": best_i,
        "best_alpha": alpha,
        "best_wsc_choice": best_choice,
        "losses": torch.tensor(losses, dtype=torch.float32),
        "s": s.to(torch.float32),
        "shift_vec": shift.to(torch.float32),
        "transform_T": T.to(torch.float32),
        "transform_T_inv": T_inv.to(torch.float32),
        "norm_comp": norm_comp,
        "clipped_weight": clipped.to(torch.float32),
        "lwc_bound": lwc_bound.to(torch.float32),
    }
