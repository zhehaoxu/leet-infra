import math

import torch


def solve(logits, k, capacity_factor):
    """位置式容量分派 + token dropping。"""
    probs = torch.softmax(logits.float(), dim=-1)
    T, E = probs.shape
    tv, ti = torch.topk(probs, k, dim=-1)
    top_w = tv / tv.sum(dim=-1, keepdim=True)
    routing_map = torch.zeros(T, E, dtype=torch.bool).scatter_(1, ti, True)
    routing_probs = torch.zeros(T, E).scatter_(1, ti, top_w)
    capacity = math.ceil(T * k / E * float(capacity_factor))

    seen = torch.zeros(E, dtype=torch.long)
    keep = torch.zeros(T, E, dtype=torch.bool)
    for t in range(T):
        for e in range(E):
            if routing_map[t, e] and int(seen[e]) < capacity:
                keep[t, e] = True
                seen[e] += 1
    final_map = routing_map & keep
    final_probs = routing_probs * final_map.float()
    dropped = int(routing_map.sum().item()) - int(final_map.sum().item())
    return final_map, final_probs, dropped
