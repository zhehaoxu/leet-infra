import math

import torch

X_T = torch.tensor([[0.4, -1.1], [0.9, 2.2]])
EPS = torch.tensor([[0.3, -0.7], [1.2, -0.4]])
AB_T = 0.65
AB_PREV = 0.72


def _oracle(x_t, eps, ab_t, ab_prev):
    """独立重算：逐元素双循环 math.sqrt。"""
    out = torch.empty_like(x_t)
    for i in range(x_t.shape[0]):
        for j in range(x_t.shape[1]):
            x0 = (x_t[i, j] - math.sqrt(1 - ab_t) * eps[i, j]) / math.sqrt(ab_t)
            out[i, j] = math.sqrt(ab_prev) * x0 + math.sqrt(1 - ab_prev) * eps[i, j]
    return out


def check_example(solve):
    y = solve(X_T, EPS, AB_T, AB_PREV)
    exp = torch.tensor([[0.39293810725212097, -1.0922671556472778],
                        [0.8350236415863037, 2.352832794189453]])
    assert torch.allclose(y, exp, atol=1e-5, rtol=1e-5), f"got {y.tolist()}"


check_example.example = True


def check_matches_oracle(solve):
    torch.manual_seed(6)
    x_t = torch.randn(3, 5)
    eps = torch.randn(3, 5)
    for ab_t, ab_prev in [(0.9, 0.95), (0.65, 0.72), (0.3, 0.5)]:
        assert torch.allclose(solve(x_t, eps, ab_t, ab_prev), _oracle(x_t, eps, ab_t, ab_prev), atol=1e-5, rtol=1e-5)


def check_identity_when_alphabars_equal(solve):
    # ab_prev == ab_t 时 DDIM 为恒等：x_prev == x_t
    y = solve(X_T, EPS, AB_T, AB_T)
    assert torch.allclose(y, X_T, atol=1e-5, rtol=1e-5), "ab_prev==ab_t 应还原 x_t"


def check_full_denoise_to_x0(solve):
    # ab_prev = 1 时 x_prev = x0_hat
    x0_hat = (X_T - math.sqrt(1 - AB_T) * EPS) / math.sqrt(AB_T)
    y = solve(X_T, EPS, AB_T, 1.0)
    assert torch.allclose(y, x0_hat, atol=1e-5, rtol=1e-5), "ab_prev=1 应等于 x0_hat"


def check_forward_inverse_consistency(solve):
    # 若 x_t 由真实 x0 前向生成，则 x0_hat 精确还原 x0，且跳到 ab_prev=1 得回 x0
    x0 = torch.tensor([[0.8, -1.3], [0.2, 1.7]])
    xt = math.sqrt(AB_T) * x0 + math.sqrt(1 - AB_T) * EPS
    y = solve(xt, EPS, AB_T, 1.0)
    assert torch.allclose(y, x0, atol=1e-4, rtol=1e-4), "确定性反演应还原生成用的 x0"
