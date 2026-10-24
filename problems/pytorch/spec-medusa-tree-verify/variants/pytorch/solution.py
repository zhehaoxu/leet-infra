import torch


def _cdf(probs, u):
    c = 0.0
    for i, p in enumerate(probs):
        c += p
        if u < c:
            return i
    return len(probs) - 1


def solve(cands, P, eps, u_bonus):
    """Medusa 多头候选树的 typical acceptance 验证。"""
    Pl = P.tolist()
    eps = float(eps)
    # 候选树节点：所有路径的全部前缀，去重
    nodes = set()
    for c in cands:
        for d in range(1, len(c) + 1):
            nodes.add(tuple(c[:d]))
    # 父链全接受的最深路径；同深取字典序最小
    best = []
    for node in nodes:
        ok = all(Pl[i - 1][node[i - 1]] >= eps for i in range(1, len(node) + 1))
        if ok:
            lst = list(node)
            if len(lst) > len(best) or (len(lst) == len(best) and lst < best):
                best = lst
    D = len(best)
    bonus = _cdf(Pl[D], float(u_bonus))
    return torch.tensor(best + [bonus], dtype=torch.long)
