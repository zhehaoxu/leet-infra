import math

import torch

_EPS = 1e-6


def _oracle(logits, k):
    """独立路径：python 手算 std/mean。"""
    probs = torch.softmax(logits.float(), dim=-1)
    T, E = probs.shape
    _, sel = torch.topk(probs, k, dim=-1)
    imp = [float(probs[:, e].sum()) for e in range(E)]
    load = [0] * E
    for t in range(T):
        for j in range(k):
            load[int(sel[t, j])] += 1

    def cv(lst):
        m = sum(lst) / len(lst)
        v = sum((x - m) ** 2 for x in lst) / len(lst)
        return math.sqrt(v) / (m + _EPS)

    return torch.tensor(cv(imp) ** 2 + cv(load) ** 2)


def check_example(solve):
    logits = torch.tensor([[2.0, 0.0, 0.0, 0.0],
                           [0.0, 2.0, 0.0, 0.0],
                           [0.0, 0.0, 2.0, 0.0],
                           [0.0, 0.0, 0.0, 2.0],
                           [1.0, 1.0, 0.0, 0.0],
                           [0.0, 1.0, 1.0, 0.0]])
    v = float(solve(logits, 1))
    assert abs(v - 0.122975) < 1e-4, v


def check_highly_unbalanced(solve):
    logits = torch.tensor([[3.0, 0.0, 0.0, 0.0],
                           [3.0, 0.0, 0.0, 0.0],
                           [3.0, 0.0, 0.0, 0.0],
                           [3.0, 0.0, 0.0, 0.0]])
    v = float(solve(logits, 1))
    assert abs(v - 5.050445) < 1e-3, v


def check_matches_oracle(solve):
    torch.manual_seed(0)
    for _ in range(30):
        T = int(torch.randint(3, 15, (1,)))
        E = int(torch.randint(3, 8, (1,)))
        k = int(torch.randint(1, E + 1, (1,)))
        logits = torch.randn(T, E)
        got = float(solve(logits, k))
        exp = float(_oracle(logits, k))
        assert abs(got - exp) < 1e-3, (T, E, k, got, exp)


def check_perfectly_balanced_is_small(solve):
    logits = torch.tensor([[3.0, 0.0, 0.0, 0.0],
                           [0.0, 3.0, 0.0, 0.0],
                           [0.0, 0.0, 3.0, 0.0],
                           [0.0, 0.0, 0.0, 3.0]])
    v = float(solve(logits, 1))
    # load 完美均衡 → cv(load) ≈ 0；importance 每行 softmax 略有余量 → cv(imp) 很小
    assert v < 0.01, v


check_example.example = True
