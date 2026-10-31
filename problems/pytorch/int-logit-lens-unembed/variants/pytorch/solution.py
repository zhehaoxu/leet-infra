import torch


def solve(hiddens, lnf_w, lnf_b, unembed, apply_norm=True, eps=1e-5):
    if not torch.is_tensor(hiddens) or hiddens.dim() != 2:
        raise ValueError("hiddens must be [L, D]")
    for name, t in (("lnf_w", lnf_w), ("lnf_b", lnf_b)):
        if not torch.is_tensor(t) or t.dim() != 1:
            raise ValueError(f"{name} must be [D]")
    if not torch.is_tensor(unembed) or unembed.dim() != 2:
        raise ValueError("unembed must be [D, V]")
    if not isinstance(apply_norm, bool):
        raise ValueError("apply_norm must be bool")
    if not isinstance(eps, (int, float)) or isinstance(eps, bool) or not (float(eps) > 0):
        raise ValueError("eps must be a positive number")
    L, D = hiddens.shape
    if lnf_w.shape != (D,) or lnf_b.shape != (D,) or unembed.shape[0] != D:
        raise ValueError("dimension mismatch")
    if L == 0 or D == 0 or unembed.shape[1] == 0:
        raise ValueError("empty input")

    hd = hiddens.double()
    wd = lnf_w.double()
    bd = lnf_b.double()
    ud = unembed.double()

    logits_rows = []
    for l in range(L):
        h = hd[l]
        if apply_norm:
            mu = h.mean()
            var = ((h - mu) ** 2).mean()  # 有偏方差
            n = (h - mu) / torch.sqrt(var + float(eps)) * wd + bd
        else:
            n = h
        logits_rows.append(n @ ud)

    def argmin_index(vals):
        mx = max(vals)
        return min(i for i, v in enumerate(vals) if v == mx)

    argmax = [argmin_index([float(x) for x in row]) for row in logits_rows]
    final_argmax = argmax[-1]
    agree = [a == final_argmax for a in argmax]
    rank_of_final = []
    top_logit = []
    for l in range(L):
        vals = [float(x) for x in logits_rows[l]]
        fv = vals[final_argmax]
        rank_of_final.append(1 + sum(1 for v in vals if v > fv))
        top_logit.append(max(vals))
    return {
        "logits": [[float(x) for x in row] for row in logits_rows],
        "argmax": argmax,
        "final_argmax": final_argmax,
        "agree": agree,
        "rank_of_final": rank_of_final,
        "top_logit": top_logit,
    }
