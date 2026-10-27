import torch


def _row_values(row):
    out = []
    for v in row:
        if v < 0:
            break
        out.append(int(v))
    return out


def solve(cache, queries, queue, alpha, beta):
    if not isinstance(cache, torch.Tensor) or cache.dim() != 2:
        raise ValueError("cache 必须是二维张量")
    if not isinstance(queries, torch.Tensor) or queries.dim() != 2:
        raise ValueError("queries 必须是二维张量")
    if not isinstance(queue, torch.Tensor) or queue.dim() != 1:
        raise ValueError("queue 必须是一维张量")
    if alpha < 0 or beta < 0:
        raise ValueError("alpha / beta 不能为负")
    n_rep = int(cache.shape[0])
    if n_rep < 1:
        raise ValueError("至少需要一个副本")
    if int(queue.numel()) != n_rep:
        raise ValueError("queue 长度必须等于副本数")

    cached = [set(_row_values(row)) for row in cache.to(torch.int64).tolist()]
    depths = queue.to(torch.int64).tolist()
    reqs = [_row_values(row) for row in queries.to(torch.int64).tolist()]

    chosen, hits = [], []
    total = 0.0
    for blocks in reqs:
        best_r, best_cost, best_hit = 0, None, 0
        for r in range(n_rep):
            matched = 0
            for v in blocks:
                if v not in cached[r]:
                    break
                matched += 1
            cost = alpha * (len(blocks) - matched) + beta * depths[r]
            if best_cost is None or cost < best_cost:
                best_r, best_cost, best_hit = r, cost, matched
        chosen.append(best_r)
        hits.append(best_hit)
        total += best_cost
        depths[best_r] += 1
        cached[best_r].update(blocks)

    return (
        torch.tensor(chosen, dtype=torch.int64),
        torch.tensor(hits, dtype=torch.int64),
        torch.tensor(depths, dtype=torch.int64),
        torch.tensor(total, dtype=torch.float32),
    )
