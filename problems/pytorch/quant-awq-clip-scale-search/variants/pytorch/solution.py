import torch


def solve(W, X, ratios, clips, max_shrink, group_size, bits):
    if bits < 2 or bits > 8:
        raise ValueError("bits must be in [2, 8]")
    Wf = W.to(torch.float64)
    Xf = X.to(torch.float64)
    O, C = Wf.shape
    qmax = 2 ** (bits - 1) - 1
    n_grp = C // group_size
    xmean = Xf.abs().mean(dim=0)
    wamax = Wf.abs().amax(dim=0)
    wmax = Wf.abs().max()
    usable = (xmean > 0) & (wamax > 0)

    losses = torch.empty(len(ratios), len(clips), dtype=torch.float64)
    best = None
    for i, beta in enumerate(ratios):
        b = float(beta)
        s = torch.ones_like(xmean)
        s[usable] = xmean[usable] ** b / wamax[usable] ** (1.0 - b)
        gmean = torch.sqrt(s.max() * s.min())
        if float(gmean) > 0:
            s = s / gmean
        s = torch.clamp(s, min=1.0 - float(max_shrink))
        for j, clip in enumerate(clips):
            c = float(clip)
            Wc = torch.clamp(Wf, -c * wmax, c * wmax)
            Wg = (Wc * s).reshape(O, n_grp, group_size)
            amax = Wg.abs().amax(dim=2, keepdim=True)
            scale = amax / qmax
            safe = torch.where(scale > 0, scale, torch.ones_like(scale))
            q = torch.clamp(torch.round(Wg / safe), -qmax, qmax)
            logical = (q * safe).reshape(O, C) / s
            loss = float((xmean * (Wc - logical) ** 2).mean())
            losses[i, j] = loss
            if best is None or loss < best[0]:
                best = (loss, b, c, s.clone(), q.reshape(O, C).clone())

    return (
        best[1],
        best[2],
        best[3].to(torch.float32),
        best[4].to(torch.int64),
        losses.to(torch.float32),
    )
