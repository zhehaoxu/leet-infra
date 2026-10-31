import math

import torch


def solve(hiddens, s, alphas, layer_indices, normalize=False):
    if not torch.is_tensor(hiddens) or hiddens.dim() != 3:
        raise ValueError("hiddens must be [L, T, D]")
    if not torch.is_tensor(s) or s.dim() != 1:
        raise ValueError("s must be [D]")
    L, T, D = hiddens.shape
    if s.shape[0] != D:
        raise ValueError("s dim mismatch")
    if D == 0 or L == 0 or T == 0:
        raise ValueError("empty input")
    if len(alphas) != len(layer_indices):
        raise ValueError("alphas and layer_indices length mismatch")
    if len(alphas) == 0:
        raise ValueError("at least one injection required")
    for a in alphas:
        if isinstance(a, bool) or not isinstance(a, (int, float)):
            raise ValueError("alpha must be a real number")
        af = float(a)
        if math.isnan(af) or math.isinf(af):
            raise ValueError("alpha must be finite")
    for idx in layer_indices:
        if not isinstance(idx, int) or isinstance(idx, bool) or not (0 <= idx < L):
            raise ValueError("layer index out of range")
    sn = float(s.double().norm())
    if sn == 0.0:
        raise ValueError("s must be nonzero")

    sd = s.double()
    s_eff = sd / sn if normalize else sd
    s_eff_norm = float(s_eff.norm())

    steered = hiddens.double().clone()
    layer_sum = [0.0] * L
    for k in range(len(alphas)):
        idx = layer_indices[k]
        a = float(alphas[k])
        steered[idx] = steered[idx] + a * s_eff
        layer_sum[idx] += a
    layer_delta = [abs(x) * s_eff_norm for x in layer_sum]

    hd = hiddens.double()
    cos_before = []
    cos_after = []
    proj_coef = []
    for l in range(L):
        cb = []
        ca = []
        pc = []
        for t in range(T):
            h = hd[l, t]
            hp = steered[l, t]
            hn = float(h.norm())
            hpn = float(hp.norm())
            dot = float(h @ sd)
            cb.append(dot / (hn * sn))
            ca.append(float(hp @ sd) / (hpn * sn))
            pc.append(dot / float(sd @ sd))
        cos_before.append(cb)
        cos_after.append(ca)
        proj_coef.append(pc)

    return {
        "steered": steered,
        "s_norm": sn,
        "layer_delta": layer_delta,
        "cos_before": cos_before,
        "cos_after": cos_after,
        "proj_coef": proj_coef,
    }
