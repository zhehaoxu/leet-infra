import torch


def solve(Q, P, tau, L):
    R, g, D, V = Q.shape
    Ql = Q.tolist()
    Pl = P.tolist()
    toks, lays, depth = [], [], []
    for r in range(R):
        if g == 0:
            toks.append(1)
            lays.append(L)
            depth.append([])
            continue
        es, ds = [], []
        for j in range(g):
            e = D
            for l in range(D):
                row = Ql[r][j][l]
                m = row[0]
                for v in range(1, V):
                    if row[v] > m:
                        m = row[v]
                if m >= tau:
                    e = l + 1
                    break
            row = Ql[r][j][e - 1]
            b, m = 0, row[0]
            for v in range(1, V):
                if row[v] > m:
                    m, b = row[v], v
            es.append(e)
            ds.append(b)
        a = 0
        for j in range(g):
            prow = Pl[r][j]
            pb, pm = 0, prow[0]
            for v in range(1, V):
                if prow[v] > pm:
                    pm, pb = prow[v], v
            if pb != ds[j]:
                break
            a += 1
        toks.append(a + 1)
        lays.append(sum(es) + max(L - min(es), 1))
        depth.append(es)
    speedup = sum(toks) * L / sum(lays)
    return (torch.tensor(toks, dtype=torch.long),
            torch.tensor(lays, dtype=torch.long),
            torch.tensor(depth, dtype=torch.long),
            float(speedup))
