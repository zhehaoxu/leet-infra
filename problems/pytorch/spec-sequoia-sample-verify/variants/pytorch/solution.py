import torch


def _norm(x):
    s = sum(x)
    if s > 0:
        return [v / s for v in x]
    n = len(x)
    return [1.0 / n] * n


def _cdf(probs, u):
    c = 0.0
    for i, p in enumerate(probs):
        c += p
        if u < c:
            return i
    return len(probs) - 1


def solve(P, Q, k, v_list, r_list, u_final):
    """Sequoia Sampling & Verification (arXiv:2402.12374 Algorithm 2)。"""
    R = _norm(P.tolist())
    D = _norm(Q.tolist())
    vl = v_list.tolist()
    rl = r_list.tolist()
    S = set()
    V = len(R)
    for i in range(int(k)):
        si = _cdf(D, float(vl[i]))
        if D[si] > 0 and float(rl[i]) < R[si] / D[si]:
            return torch.tensor([si, 0], dtype=torch.long)
        R = _norm([max(0.0, R[v] - D[v]) for v in range(V)])
        D[si] = 0.0
        S.add(si)
        if sum(D) == 0:
            D = [0.0 if t in S else 1.0 for t in range(V)]
        D = _norm(D)
    return torch.tensor([_cdf(R, float(u_final)), 1], dtype=torch.long)
