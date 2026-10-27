import math

import torch


def _lam(ab):
    return 0.5 * math.log(ab / (1.0 - ab))


def _oracle(x, d0, ab_t, ab_s0, d1=None, ab_s1=None):
    """独立 oracle：逐元素 math 重算（sigmoid 化的 λ 写法，与参考实现的 log(α/σ) 不同源）。"""
    h = _lam(ab_t) - _lam(ab_s0)
    coef = math.sqrt(ab_t) * (1.0 - math.exp(-h))
    scale = math.sqrt((1.0 - ab_t) / (1.0 - ab_s0))
    total = d0
    if d1 is not None:
        total = d0 + 0.5 * (d0 - d1) / (_lam(ab_s0) - _lam(ab_s1))
    return scale * x + coef * total


_AB_T, _AB_S0, _AB_S1 = 0.81, 0.64, 0.49


def check_example(solve):
    got1 = solve(torch.tensor([1.0]), torch.tensor([0.5]), _AB_T, _AB_S0, order=1)
    assert abs(float(got1[0]) - 0.885890) < 1e-5, f"Euler 步应为 0.885890，got {float(got1[0])}"

    got2 = solve(
        torch.tensor([1.0]),
        torch.tensor([0.5]),
        _AB_T,
        _AB_S0,
        order=2,
        x0_pred_prev=torch.tensor([0.4]),
        alpha_bar_s1=_AB_S1,
    )
    assert abs(float(got2[0]) - 0.937698) < 1e-5, f"2M 步应为 0.937698，got {float(got2[0])}"


check_example.example = True


def check_first_order_is_ddim(solve):
    """data-prediction 的 Euler 步与 DDIM 闭式完全等价（同一条指数积分器的两种写法）。"""
    torch.manual_seed(0)
    x = torch.randn(4, 6, dtype=torch.float64)
    d0 = torch.randn(4, 6, dtype=torch.float64)
    for ab_t, ab_s0 in ((0.81, 0.64), (0.5, 0.2), (0.99, 0.9), (0.04, 0.01)):
        got = solve(x, d0, ab_t, ab_s0, order=1)
        a_t, s_t = math.sqrt(ab_t), math.sqrt(1.0 - ab_t)
        a_s, s_s = math.sqrt(ab_s0), math.sqrt(1.0 - ab_s0)
        ddim = a_t * d0 + s_t * (x - a_s * d0) / s_s
        assert torch.allclose(got, ddim.to(torch.float32), atol=1e-5), f"ᾱ=({ab_t},{ab_s0}) 与 DDIM 不一致"


def check_second_order_correction(solve):
    """二阶只是在 D0 上加了 0.5·D1 的泰勒项：差分可独立复算。"""
    torch.manual_seed(1)
    x = torch.randn(3, 4, dtype=torch.float64)
    d0 = torch.randn(3, 4, dtype=torch.float64)
    d1 = torch.randn(3, 4, dtype=torch.float64)

    one = solve(x, d0, _AB_T, _AB_S0, order=1)
    two = solve(x, d0, _AB_T, _AB_S0, order=2, x0_pred_prev=d1, alpha_bar_s1=_AB_S1)

    h = _lam(_AB_T) - _lam(_AB_S0)
    r0 = _lam(_AB_S0) - _lam(_AB_S1)
    coef = math.sqrt(_AB_T) * (1.0 - math.exp(-h))
    want_delta = coef * 0.5 * (d0 - d1) / r0
    assert torch.allclose((two - one).to(torch.float64), want_delta, atol=1e-5), "二阶修正项系数不对"

    # 两次预测相同时，二阶退化为一阶（D1 = 0）
    same = solve(x, d0, _AB_T, _AB_S0, order=2, x0_pred_prev=d0, alpha_bar_s1=_AB_S1)
    assert torch.allclose(same, one, atol=1e-6), "预测不变时二阶应退化为一阶"


def check_limits(solve):
    torch.manual_seed(2)
    x = torch.randn(2, 5, dtype=torch.float64)
    d0 = torch.randn(2, 5, dtype=torch.float64)

    # h = 0：不前进
    same_t = solve(x, d0, 0.64, 0.64, order=1)
    assert torch.allclose(same_t, x.to(torch.float32), atol=1e-5), "ᾱ 不变时应原样返回"

    # ᾱ_t -> 1：σ_t -> 0 且 h -> ∞，结果收敛到 x0 预测（σ_t/σ_s0 只剩极小残差）
    near_clean = solve(x, d0, 0.9999999999, 0.5, order=1)
    assert torch.allclose(near_clean, d0.to(torch.float32), atol=1e-3), "走到最干净处应等于 x0 预测"

    # 中间步必然落在 x 与 x0 预测之间（coef 与 σ 比都为正）
    mid = solve(x, d0, 0.81, 0.64, order=1).to(torch.float64)
    assert torch.all((mid - d0).abs() <= (x - d0).abs() + 1e-6), "一步不应越过 x0 预测"


def check_random_and_invalid(solve):
    torch.manual_seed(3)
    x = torch.randn(3, 2, 2, dtype=torch.float64)
    d0 = torch.randn(3, 2, 2, dtype=torch.float64) * 3
    d1 = torch.randn(3, 2, 2, dtype=torch.float64) * 3
    for ab_t, ab_s0, ab_s1 in ((0.81, 0.64, 0.49), (0.3, 0.2, 0.1), (0.06, 0.04, 0.01)):
        got1 = solve(x, d0, ab_t, ab_s0, order=1)
        want1 = _oracle(x, d0, ab_t, ab_s0)
        assert torch.allclose(got1, want1.to(torch.float32), atol=1e-5), "一阶与 oracle 不一致"
        got2 = solve(x, d0, ab_t, ab_s0, order=2, x0_pred_prev=d1, alpha_bar_s1=ab_s1)
        want2 = _oracle(x, d0, ab_t, ab_s0, d1, ab_s1)
        assert torch.allclose(got2, want2.to(torch.float32), atol=1e-5), "二阶与 oracle 不一致"

    good = torch.zeros(2)
    for bad in (
        lambda: solve([0.0, 1.0], good, 0.8, 0.6),
        lambda: solve(good, torch.zeros(3), 0.8, 0.6),
        lambda: solve(good, good, 0.8, 0.6, order=3),
        lambda: solve(good, good, 0.0, 0.6),
        lambda: solve(good, good, 0.8, 1.0),
        lambda: solve(good, good, 0.8, 0.6, order=2),
        lambda: solve(good, good, 0.8, 0.6, order=2, x0_pred_prev=good),
        # s1 比 s0 更干净：λ_s0 - λ_s1 <= 0
        lambda: solve(good, good, 0.8, 0.6, order=2, x0_pred_prev=good, alpha_bar_s1=0.9),
    ):
        try:
            bad()
            raise AssertionError("非法输入应抛 ValueError")
        except ValueError:
            pass
