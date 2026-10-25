import itertools
import math

import torch

N, C, K = 10, 2, 3
EXAMPLE_VALUE = 0.5333333333  # 参考解跑出的 pass@3（= 1 - 56/120）


def _oracle_bruteforce(n, c, k):
    """独立重算：枚举全部 C(n,k) 个 k-子集，数「至少含一个正确样本」的比例（纯定义法）。"""
    total = 0
    good = 0
    for sub in itertools.combinations(range(n), k):
        total += 1
        if any(i < c for i in sub):      # 约定前 c 个索引是正确样本，不失一般性
            good += 1
    return good / total


def _oracle_product(n, c, k):
    """再一路：把 C(n-c,k)/C(n,k) 写成 k 个连乘因子逐项相乘（无组合数、无枚举）。"""
    p = 1.0
    for i in range(k):
        if n - c - i <= 0:
            p = 0.0
            break
        p *= (n - c - i) / (n - i)
    return 1.0 - p


def _close(a, b):
    return torch.allclose(
        torch.tensor(float(a), dtype=torch.float64),
        torch.tensor(float(b), dtype=torch.float64),
        atol=1e-5,
        rtol=1e-5,
    )


def check_example(solve):
    got = solve(N, C, K)
    assert _close(got, EXAMPLE_VALUE), f"got {got}"
    assert _close(got, 1.0 - 56.0 / 120.0), got     # C(8,3)/C(10,3) = 56/120


check_example.example = True


def check_matches_bruteforce_definition(solve):
    for n in range(1, 8):
        for c in range(0, n + 1):
            for k in range(1, n + 1):
                got = solve(n, c, k)
                exp = _oracle_bruteforce(n, c, k)
                assert _close(got, exp), (n, c, k, got, exp)


def check_matches_product_path(solve):
    for n, c, k in [(50, 5, 7), (100, 10, 5), (64, 1, 16), (32, 31, 8), (200, 40, 12)]:
        got = solve(n, c, k)
        assert _close(got, _oracle_product(n, c, k)), (n, c, k, got)


def check_zero_correct_is_zero(solve):
    # c = 0：一个都没通过，任何 k 都是 0
    for n in (1, 5, 20, 64):
        for k in range(1, n + 1):
            assert solve(n, 0, k) == 0.0, (n, k)


def check_k_equals_one_is_sample_rate(solve):
    # k = 1 必须退化成经验通过率 c/n（这正是无偏估计量的动机）
    for n, c in [(10, 1), (10, 3), (128, 17), (64, 64)]:
        assert _close(solve(n, c, 1), c / n), (n, c, solve(n, c, 1))


def check_k_equals_n_is_all_or_nothing(solve):
    # k = n：取遍全部样本，只要有一个正确就必为 1，否则为 0
    assert solve(7, 0, 7) == 0.0
    assert solve(7, 1, 7) == 1.0
    assert solve(7, 7, 7) == 1.0


def check_all_correct_is_one(solve):
    # c = n：全对 -> 恒为 1
    for k in range(1, 6):
        assert solve(5, 5, k) == 1.0, k


def check_guaranteed_hit_when_wrong_too_few(solve):
    # n - c < k：错的样本不够填满 k 个，任取 k 个必含正确样本 -> 恰为 1.0
    n, c = 10, 6
    for k in range(5, 11):          # n - c = 4，故 k >= 5 时恒 1
        assert solve(n, c, k) == 1.0, (k, solve(n, c, k))
    assert solve(n, c, 4) < 1.0     # k = 4 还能全避开正确样本


def check_monotone_non_decreasing_in_k(solve):
    prev = -1.0
    for k in range(1, 13):
        cur = solve(12, 3, k)
        assert cur >= prev, (k, cur, prev)
        prev = cur
    assert prev == 1.0               # k = n 时封顶


def check_exact_rational_value(solve):
    # n=5, c=1, k=2：1 - C(4,2)/C(5,2) = 1 - 6/10 = 0.4（整数运算保证不被浮点比值污染）
    assert _close(solve(5, 1, 2), 0.4), solve(5, 1, 2)
    assert solve(5, 1, 2) == 1.0 - math.comb(4, 2) / math.comb(5, 2)


def check_biased_estimator_disagrees(solve):
    # 常见错误写法 1 - (1 - c/n)**k 是**有偏**的，且系统性低估无偏值（k>1 时）
    unbiased = solve(10, 2, 3)
    biased = 1.0 - (1.0 - 2.0 / 10.0) ** 3
    assert abs(unbiased - (1.0 - 56.0 / 120.0)) < 1e-12, unbiased
    assert abs(biased - 0.488) < 1e-12, biased
    assert unbiased > biased, (unbiased, biased)
    # k = 1 时两者重合，都等于 c/n
    assert _close(solve(10, 2, 1), 1.0 - (1.0 - 2.0 / 10.0)), solve(10, 2, 1)
