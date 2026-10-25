import torch


def _oracle(pi, pf, cur, first, final):
    if cur <= first:
        p = pi
    elif cur >= final:
        p = pf
    else:
        frac = (cur - first) / (final - first)
        p = pf + (pi - pf) * (1.0 - frac) ** 3
    return p


def check_example(solve):
    y = solve(0.0, 0.8, 500, 0, 1000)
    assert abs(y.item() - 0.699999988079071) < 1e-5, f"got {y.item()}"


check_example.example = True


def check_matches_oracle(solve):
    for pi, pf, cur, first, final in [
        (0.0, 0.8, 0, 0, 1000),
        (0.0, 0.8, 1000, 0, 1000),
        (0.0, 0.8, 250, 0, 1000),
        (0.1, 0.9, 400, 100, 900),
        (0.5, 0.5, 333, 0, 1000),
        (0.0, 0.95, 50, 0, 1000),
    ]:
        assert abs(solve(pi, pf, cur, first, final).item() - _oracle(pi, pf, cur, first, final)) < 1e-5


def check_boundaries(solve):
    # t=t0 → p_i, t=t1 → p_f
    assert abs(solve(0.1, 0.9, 100, 100, 500).item() - 0.1) < 1e-6
    assert abs(solve(0.1, 0.9, 500, 100, 500).item() - 0.9) < 1e-6
    # 窗口外钳位
    assert abs(solve(0.1, 0.9, 0, 100, 500).item() - 0.1) < 1e-6
    assert abs(solve(0.1, 0.9, 999, 100, 500).item() - 0.9) < 1e-6


def check_constant_when_equal(solve):
    # p_i == p_f 时任意步都等于该值
    for cur in [0, 250, 777, 1000]:
        assert abs(solve(0.6, 0.6, cur, 0, 1000).item() - 0.6) < 1e-6


def check_monotone_and_cubic(solve):
    # 三次（非线形）：中点值 != 线性中点；且随 t 单调
    mid = solve(0.0, 0.8, 500, 0, 1000).item()
    linear_mid = 0.4
    assert abs(mid - linear_mid) > 0.1, "三次调度中点应显著偏离线性中点"
    ys = [solve(0.0, 0.8, t, 0, 1000).item() for t in [0, 250, 500, 750, 1000]]
    assert all(ys[i] <= ys[i + 1] + 1e-9 for i in range(len(ys) - 1)), ys
