import random

import numpy as np
import torch


def _oracle(alphas, budget):
    """独立 oracle：暴力枚举所有 γ 组合，返回最大期望吐词总数。"""
    a = alphas.numpy().astype(np.float64)
    n = a.size
    best = -1.0

    def rec(i, left, acc):
        nonlocal best
        if left < 0:
            return
        if i == n:
            best = max(best, acc)
            return
        g = 0
        while g + 1 <= left:
            rec(i + 1, left - (g + 1), acc + sum(a[i] ** k for k in range(g + 1)))
            g += 1

    rec(0, int(budget), 0.0)
    return best


def check_example(solve):
    torch.manual_seed(0)
    gammas, expected, total = solve(torch.tensor([0.8, 0.5, 0.2]), 5)
    assert gammas.tolist() == [2, 0, 0], f"γ 应为 [2, 0, 0]，got {gammas.tolist()}"
    want = torch.tensor([2.44, 1.0, 1.0])
    assert torch.allclose(expected, want, atol=1e-6), f"got {expected.tolist()}"
    assert abs(float(total) - 4.44) < 1e-6, f"总期望应为 4.44，got {float(total)}"


check_example.example = True


def check_exact_budget_gives_zero_gamma(solve):
    """名额刚好等于批大小：每个请求只拿自己那一列，全部 γ=0。"""
    torch.manual_seed(1)
    alphas = torch.tensor([0.9, 0.6, 0.3, 0.1, 0.99])
    gammas, expected, total = solve(alphas, 5)
    assert gammas.tolist() == [0] * 5, f"got {gammas.tolist()}"
    assert torch.allclose(expected, torch.ones(5), atol=1e-6), "γ=0 时期望恒为 1"
    assert abs(float(total) - 5.0) < 1e-6, f"got {float(total)}"


def check_ties_go_to_smallest_id(solve):
    """边际收益并列时给下标最小的请求；α=1 时逐位收益恒为 1。"""
    torch.manual_seed(2)
    # 三个请求 α 相同：第一个富余名额给 r0，第二个名额则并列在 r1/r2 之间 -> 给 r1
    gammas, expected, total = solve(torch.full((3,), 0.5), 5)
    assert gammas.tolist() == [1, 1, 0], f"got {gammas.tolist()}"
    assert torch.allclose(
        expected, torch.tensor([1.5, 1.5, 1.0]), atol=1e-6
    ), f"got {expected.tolist()}"
    assert abs(float(total) - 4.0) < 1e-6, f"got {float(total)}"

    gammas, expected, total = solve(torch.ones(2), 6)
    assert gammas.tolist() == [4, 0], f"α=1 时同样并列取小下标，got {gammas.tolist()}"
    assert torch.allclose(expected, torch.tensor([5.0, 1.0]), atol=1e-6)
    assert abs(float(total) - 6.0) < 1e-6, "α=1 时期望吐词等于名额本身"


def check_random_against_bruteforce(solve):
    """随机批与暴力枚举 oracle 对拍，并核对名额不超发。"""
    rng = random.Random(31)
    for _ in range(40):
        n = rng.randint(1, 3)
        alphas = torch.tensor([round(rng.uniform(0.05, 0.95), 3) for _ in range(n)])
        budget = rng.randint(n, n + 6)
        gammas, expected, total = solve(alphas, budget)
        assert int((gammas + 1).sum()) <= budget, "名额超发了"
        assert gammas.tolist() == [] or int(gammas.min()) >= 0, f"got {gammas.tolist()}"
        want = _oracle(alphas, budget)
        assert abs(float(total) - want) < 1e-5, f"总期望 {float(total)} vs oracle {want}"
        assert abs(float(expected.sum()) - float(total)) < 1e-5, "总和应与逐项一致"


def check_invalid_inputs(solve):
    for bad in (
        lambda: solve(torch.tensor([]), 3),
        lambda: solve(torch.tensor([0.5, 1.2]), 4),
        lambda: solve(torch.tensor([-0.1, 0.5]), 4),
        lambda: solve(torch.tensor([0.5, 0.5]), 1),
        lambda: solve(torch.zeros((2, 2)), 4),
    ):
        try:
            bad()
            raise AssertionError("非法输入应抛 ValueError")
        except ValueError:
            pass
