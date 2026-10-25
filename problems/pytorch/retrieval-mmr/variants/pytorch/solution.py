"""参考答案：归一化后矩阵化算余弦，再 k 轮贪心 argmax（升序+容差+并列取小）。"""

import torch


def solve(query, docs, lam, k):
    D = docs.to(torch.float64)
    q = query.to(torch.float64).view(1, -1)

    def unit(m):
        nrm = m.norm(dim=1, keepdim=True)
        nrm = torch.where(nrm == 0, torch.ones_like(nrm), nrm)
        return m / nrm

    Du = unit(D)
    qu = unit(q)
    rel = (Du @ qu.T).view(-1)          # (n,) cos(doc, query)
    sim = Du @ Du.T                       # (n, n) cos(doc_i, doc_j)

    n = D.shape[0]
    selected = []
    remaining = list(range(n))
    for _ in range(k):
        best_i, best_v = None, None
        for i in remaining:
            pen = max((float(sim[i, s]) for s in selected), default=0.0)
            v = lam * float(rel[i]) - (1 - lam) * pen
            if best_v is None or v > best_v + 1e-12:
                best_v, best_i = v, i
        selected.append(best_i)
        remaining.remove(best_i)
    return torch.tensor(selected, dtype=torch.long)
