import torch


def _cdf(row, u):
    s = 0.0
    for j, v in enumerate(row):
        s += v
        if s > u:
            return j
    return len(row) - 1


def solve(P, Q, cand, u, u_end, no_replace):
    Pl = P.tolist()
    Ql = Q.tolist()
    Cl = cand.tolist()
    Ul = u.tolist()
    g = len(Ql)
    V = P.shape[1]
    C = len(Cl[0]) if g else 0
    out, picks = [], []
    trials = 0
    for j in range(g):
        p = Pl[j][:]
        q = Ql[j][:]
        hit = -1
        for i in range(C):
            x = Cl[j][i]
            px, qx = p[x], q[x]
            if qx > 0.0:
                ratio = px / qx
                if ratio > 1.0:
                    ratio = 1.0
            elif px > 0.0:
                ratio = float("inf")
            else:
                ratio = 0.0
            trials += 1
            if Ul[j][i] <= ratio:
                hit = i
                break
            res = [0.0] * V
            s = 0.0
            for v in range(V):
                d = p[v] - q[v]
                if d > 0.0:
                    res[v] = d
                s += res[v]
            if s > 0.0:
                p = [res[v] / s for v in range(V)]
            if no_replace:
                q[x] = 0.0
                s2 = 0.0
                for v in range(V):
                    s2 += q[v]
                if s2 > 0.0:
                    q = [q[v] / s2 for v in range(V)]
        if hit < 0:
            picks.append(-1)
            out.append(_cdf(p, u_end))
            break
        picks.append(hit)
        out.append(x)
    else:
        out.append(_cdf(Pl[g], u_end))
    return (torch.tensor(out, dtype=torch.long),
            torch.tensor(picks, dtype=torch.long),
            int(trials))
