import torch


def _topk_ids(values, k):
    if k <= 0:
        return []
    return torch.argsort(-values, stable=True).tolist()[:k]


def solve(scores, budget, edges):
    """CAKE 预算分配 + 跨层级联（float64 计算、int64/bool 输出）。"""
    layers, length = scores.shape
    budget = int(budget)
    if layers < 1 or length < 1:
        raise ValueError("scores 至少为 (1,1)")
    if budget < 0:
        raise ValueError("budget 必须 >= 0")
    if bool((scores < 0).any()):
        raise ValueError("scores 必须非负")
    for e in edges:
        if len(e) != 2:
            raise ValueError("edges 每项必须是 (src, dst)")
        src, dst = int(e[0]), int(e[1])
        if not (0 <= src < layers and 0 <= dst < layers):
            raise ValueError("层号越界")
        if src == dst:
            raise ValueError("不允许自环")

    s64 = scores.to(torch.float64)
    mass = s64.sum(dim=1)
    total = float(mass.sum())
    if total > 0:
        raw = budget * mass / total
    else:
        raw = torch.zeros(layers, dtype=torch.float64)
    alloc = torch.clamp(torch.floor(raw).to(torch.long), max=length)

    rem = budget - int(alloc.sum())
    if rem > 0:
        frac = (raw - torch.floor(raw)).tolist()
        order = sorted(range(layers), key=lambda l: (-frac[l], l))
        while rem > 0:
            progressed = False
            for l in order:
                if rem == 0:
                    break
                if int(alloc[l]) < length:
                    alloc[l] += 1
                    rem -= 1
                    progressed = True
            if not progressed:
                break

    work = s64.clone()
    for src, dst in edges:
        work[int(dst)] = work[int(dst)] + work[int(src)]

    keep = torch.zeros(layers, length, dtype=torch.bool)
    for l in range(layers):
        picked = _topk_ids(work[l], int(alloc[l]))
        if picked:
            keep[l, picked] = True
    return alloc, keep
