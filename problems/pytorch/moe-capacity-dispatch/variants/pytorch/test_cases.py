import math

import torch


def _oracle(logits, k, capacity_factor):
    """独立：显式 per-expert 队列 + 遍历。"""
    probs = torch.softmax(logits.float(), dim=-1)
    T, E = probs.shape
    tv, ti = torch.topk(probs, k, dim=-1)
    top_w = tv / tv.sum(dim=-1, keepdim=True)
    capacity = math.ceil(T * k / E * float(capacity_factor))
    per_expert = [[] for _ in range(E)]
    for t in range(T):
        for j in range(k):
            per_expert[int(ti[t, j])].append((t, float(top_w[t, j])))
    final_map = torch.zeros(T, E, dtype=torch.bool)
    final_probs = torch.zeros(T, E)
    kept = 0
    for e in range(E):
        for (t, w) in per_expert[e][:capacity]:
            final_map[t, e] = True
            final_probs[t, e] = w
            kept += 1
    dropped = T * k - kept
    routing_probs_full = torch.zeros(T, E).scatter_(1, ti, top_w)
    return final_map, final_probs, dropped, routing_probs_full


def check_example(solve):
    logits = torch.tensor([[2.0, 0, 0, 0], [1.0, 0, 0, 0], [0, 2.0, 0, 0], [0, 3.0, 0, 0],
                           [0, 0, 2.0, 0], [0, 0, 1.0, 0], [0, 0, 0, 2.0], [0, 0, 0, 3.0]])
    fm, fp, dr = solve(logits, 1, 0.5)
    exp = torch.tensor([[1, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0],
                        [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]], dtype=torch.bool)
    assert (fm == exp).all(), fm.int().tolist()
    assert dr == 4, dr


def check_capacity_one_per_expert(solve):
    logits = torch.tensor([[2.0, 0, 0, 0], [1.0, 0, 0, 0], [0, 2.0, 0, 0], [0, 3.0, 0, 0],
                           [0, 0, 2.0, 0], [0, 0, 1.0, 0], [0, 0, 0, 2.0], [0, 0, 0, 3.0]])
    fm, _, _ = solve(logits, 1, 0.5)
    for e in range(4):
        assert int(fm[:, e].sum()) == 1


def check_no_drop_when_capacity_generous(solve):
    torch.manual_seed(3)
    logits = torch.randn(10, 4)
    fm, _, dr = solve(logits, 1, 2.0)  # C=ceil(10*1/4*2)=5, plenty
    assert dr == 0, dr
    assert int(fm.sum()) == 10


def check_matches_oracle(solve):
    torch.manual_seed(0)
    for _ in range(30):
        T = int(torch.randint(4, 15, (1,)))
        E = int(torch.randint(3, 7, (1,)))
        k = int(torch.randint(1, min(3, E) + 1, (1,)))
        cf = [0.5, 1.0, 1.25, 2.0][int(torch.randint(0, 4, (1,)))]
        logits = torch.randn(T, E)
        m1, p1, d1 = solve(logits, k, cf)
        m2, p2, d2, _ = _oracle(logits, k, cf)
        assert (m1 == m2).all(), (T, E, k, cf, d1, d2)
        assert torch.allclose(p1, p2, atol=1e-6)
        assert d1 == d2, (d1, d2)


def check_probs_zero_where_dropped(solve):
    logits = torch.tensor([[2.0, 0, 0, 0], [1.0, 0, 0, 0], [0, 2.0, 0, 0], [0, 3.0, 0, 0],
                           [0, 0, 2.0, 0], [0, 0, 1.0, 0], [0, 0, 0, 2.0], [0, 0, 0, 3.0]])
    fm, fp, _ = solve(logits, 1, 0.5)
    assert bool(((fp > 0) == fm).all())


check_example.example = True
