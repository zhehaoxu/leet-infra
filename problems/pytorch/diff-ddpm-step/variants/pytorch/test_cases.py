import math

import torch

X_T = torch.tensor([[0.4, -1.1], [0.9, 2.2]])
X0P = torch.tensor([[0.7, -1.5], [1.1, 2.6]])
BETA_T = 0.02
AB_T = 0.65
AB_PREV = 0.72
Z = torch.tensor([[0.2, -0.3], [0.5, 0.1]])


def _coeffs(beta_t, ab_t, ab_prev):
    a_t = 1 - beta_t
    denom = 1 - ab_t
    c_t = math.sqrt(a_t) * (1 - ab_prev) / denom
    c_0 = math.sqrt(ab_prev) * beta_t / denom
    var = beta_t * (1 - ab_prev) / denom
    return c_t, c_0, var


def _oracle(x_t, x0p, beta_t, ab_t, ab_prev, z):
    """独立重算：逐元素双循环用 math 重算同式（结构异于参考解的向量化）。"""
    c_t, c_0, var = _coeffs(beta_t, ab_t, ab_prev)
    sv = math.sqrt(var)
    out = torch.empty_like(x_t)
    for i in range(x_t.shape[0]):
        for j in range(x_t.shape[1]):
            out[i, j] = c_t * x_t[i, j] + c_0 * x0p[i, j] + sv * z[i, j]
    return out


def check_example(solve):
    y = solve(X_T, X0P, BETA_T, AB_T, AB_PREV, Z)
    exp = torch.tensor([[0.37602317333221436, -0.9818338751792908],
                        [0.8293452262878418, 1.8810272216796875]])
    assert torch.allclose(y, exp, atol=1e-5, rtol=1e-5), f"got {y.tolist()}"


check_example.example = True


def check_matches_oracle(solve):
    torch.manual_seed(4)
    x_t = torch.randn(3, 4)
    x0p = torch.randn(3, 4)
    z = torch.randn(3, 4)
    for beta_t, ab_t, ab_prev in [(0.01, 0.9, 0.95), (0.02, 0.65, 0.72), (0.15, 0.3, 0.4)]:
        y = solve(x_t, x0p, beta_t, ab_t, ab_prev, z)
        assert torch.allclose(y, _oracle(x_t, x0p, beta_t, ab_t, ab_prev, z), atol=1e-5, rtol=1e-5), (beta_t, ab_t)


def check_zero_z_is_mean(solve):
    # z=0 时输出等于后验均值 mu_theta = c_t*x_t + c_0*x0p
    c_t, c_0, _ = _coeffs(BETA_T, AB_T, AB_PREV)
    mean = c_t * X_T + c_0 * X0P
    y = solve(X_T, X0P, BETA_T, AB_T, AB_PREV, torch.zeros_like(Z))
    assert torch.allclose(y, mean, atol=1e-6, rtol=1e-6)


def check_noise_term_magnitude(solve):
    # 噪声项幅度 = sqrt(var)*z：用 x_t=x0p=0 探针，输出应恰为 sqrt(var)*z
    c_t, c_0, var = _coeffs(BETA_T, AB_T, AB_PREV)
    z = torch.ones(2, 2)
    y = solve(torch.zeros(2, 2), torch.zeros(2, 2), BETA_T, AB_T, AB_PREV, z)
    assert torch.allclose(y, math.sqrt(var) * z, atol=1e-6, rtol=1e-6), "零信号时仅剩 sqrt(var)*z"


def check_shape_dtype(solve):
    y = solve(X_T, X0P, BETA_T, AB_T, AB_PREV, Z)
    assert y.shape == X_T.shape and y.dtype == torch.float32
