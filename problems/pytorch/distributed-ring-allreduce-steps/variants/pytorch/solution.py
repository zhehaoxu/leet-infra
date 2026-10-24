import torch


def solve(init: torch.Tensor) -> list[torch.Tensor]:
    W = init.shape[0]
    bufs = [init[q].clone() for q in range(W)]   # 每 rank 一行，独立副本
    snaps: list[torch.Tensor] = []

    # 阶段 1 · Reduce-Scatter：rank q 发下标 (q-1-t)%W 给右邻居，接收方累加同下标
    for t in range(W - 1):
        sends = {}                                # right -> (idx, value)，用更新前缓冲
        for q in range(W):
            s = (q - 1 - t) % W
            sends[(q + 1) % W] = (s, bufs[q][s].clone())
        for r, (s, val) in sends.items():
            bufs[r][s] = bufs[r][s] + val
        snaps.append(torch.stack([b.clone() for b in bufs], dim=0))

    # 阶段 2 · All-Gather：rank q 发下标 (q-t)%W 给右邻居，接收方覆盖同下标
    for t in range(W - 1):
        sends = {}
        for q in range(W):
            s = (q - t) % W
            sends[(q + 1) % W] = (s, bufs[q][s].clone())
        for r, (s, val) in sends.items():
            bufs[r][s] = val
        snaps.append(torch.stack([b.clone() for b in bufs], dim=0))

    return snaps
