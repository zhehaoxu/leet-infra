import torch


def _cdf(probs, u):
    c = 0.0
    for i, p in enumerate(probs):
        c += p
        if u < c:
            return i
    return len(probs) - 1


def solve(P, draft, eps, u_new):
    """Typical acceptance 投机解码验证。"""
    Pl = P.tolist()
    gamma = int(draft.shape[0]) if hasattr(draft, "shape") else len(draft)
    eps = float(eps)
    out = []
    for t in range(gamma):
        x = int(draft[t])
        if Pl[t][x] >= eps:
            out.append(x)
        else:
            out.append(_cdf(Pl[t], float(u_new)))
            return torch.tensor(out, dtype=torch.long)
    out.append(_cdf(Pl[gamma], float(u_new)))
    return torch.tensor(out, dtype=torch.long)
