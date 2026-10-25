"""参考答案：每轮基于当前点重算距离，跳到「更近邻居中最近者」直到局部最优。"""

import torch


def solve(points, adj, entry, query):
    P = points.to(torch.float64)
    q = query.to(torch.float64)

    def dist2(i):
        d = P[i] - q
        return float(d @ d)

    cur = int(entry)
    while True:
        cd = dist2(cur)
        best, bd = None, cd
        for nb in adj[cur].tolist():
            nb = int(nb)
            if nb < 0:
                continue
            dn = dist2(nb)
            if dn < cd and (best is None or dn < bd or (dn == bd and nb < best)):
                best, bd = nb, dn
        if best is None:
            return torch.tensor(cur, dtype=torch.long)
        cur = best
