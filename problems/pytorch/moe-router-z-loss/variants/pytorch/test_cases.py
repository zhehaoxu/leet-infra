import math

import torch


def _oracle(logits):
    """独立：python 手算 max+log(sum(exp(x-max)))。"""
    T, E = logits.shape
    total = 0.0
    for t in range(T):
        row = [float(logits[t, e]) for e in range(E)]
        mx = max(row)
        s = sum(math.exp(x - mx) for x in row)
        ls = mx + math.log(s)
        total += ls * ls
    return torch.tensor(total / T)


def check_example(solve):
    logits = torch.tensor([[1.0, 2.0, 0.0, -1.0],
                           [0.5, 0.5, 0.5, 0.5],
                           [3.0, 0.0, 0.0, 0.0]])
    v = float(solve(logits))
    assert abs(v - 6.455749) < 1e-4, v


def check_matches_oracle(solve):
    torch.manual_seed(0)
    for _ in range(30):
        T = int(torch.randint(1, 10, (1,)))
        E = int(torch.randint(2, 12, (1,)))
        logits = torch.randn(T, E)
        assert abs(float(solve(logits)) - float(_oracle(logits))) < 1e-4


def check_translation_invariance(solve):
    # 所有 logits 加常数 c → logsumexp 增加 c → z-loss 变化 = (ls+c)^2 平均
    logits = torch.tensor([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]])
    base = solve(logits)
    shifted = solve(logits + 10.0)
    # 每行原 ls + 10 → 平方后差 (10+2*ls)*10
    ls = torch.logsumexp(logits, dim=-1)
    expect_delta = ((ls + 10) ** 2 - ls ** 2).mean()
    assert abs(float(shifted - base) - float(expect_delta)) < 1e-3


def check_all_zeros(solve):
    # 每行 [0, 0, 0, 0] → logsumexp = log(4) ≈ 1.386294，平方 ≈ 1.921812
    logits = torch.zeros(3, 4)
    v = float(solve(logits))
    assert abs(v - 1.921812) < 1e-5, v


check_example.example = True
