"""独立 oracle：按 2x2 块逐个填 R，再用矩阵乘法得 y，不复用参考实现的逐元素公式。"""

import math
import random

import torch


def _cols(theta, alpha, variant, group_size, out_features):
    n_group = out_features // group_size
    half = group_size // 2
    first = [0.0] * out_features
    second = [0.0] * out_features
    for g in range(n_group):
        for j in range(group_size):
            k = g * group_size + j
            if variant == "road_1":
                idx = g * half + (j % half)
                first[k] = alpha[idx] * math.cos(theta[idx])
                second[k] = alpha[idx] * math.sin(theta[idx])
            elif variant == "road_2":
                first[k] = alpha[k] * math.cos(theta[k])
                second[k] = alpha[k] * math.sin(theta[k])
            else:
                idx = 2 * g * group_size
                first[k] = alpha[idx + j] * math.cos(theta[idx + j])
                second[k] = alpha[idx + group_size + j] * math.sin(theta[idx + group_size + j])
    return first, second


def _oracle(theta, alpha, variant, group_size, out_features, x):
    first, second = _cols(theta, alpha, variant, group_size, out_features)
    half = group_size // 2
    R = torch.zeros(out_features, out_features, dtype=torch.float64)
    for g in range(out_features // group_size):
        for j in range(half):
            a = g * group_size + j
            b = a + half
            R[a, a] = first[a]
            R[b, b] = first[b]
            R[a, b] = -second[a]
            R[b, a] = second[b]
    xf = x.to(torch.float64)
    y = xf @ R.T
    return R.float(), y.float()


def check_example(solve):
    """road_1：group=4 时第 j 个元素与第 j+2 个配对，两对各自一个角。"""
    theta = torch.tensor([0.1, 0.2, 0.3, 0.4])
    alpha = torch.ones(4)
    x = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    r, y = solve(theta, alpha, "road_1", 4, 8, x)
    assert r.shape == (8, 8) and y.shape == x.shape, (r.shape, y.shape)
    block = r[:4, :4]
    want = torch.tensor(
        [
            [0.9950042, 0.0, -0.0998334, 0.0],
            [0.0, 0.9800666, 0.0, -0.1986693],
            [0.0998334, 0.0, 0.9950042, 0.0],
            [0.0, 0.1986693, 0.0, 0.9800666],
        ]
    )
    assert torch.allclose(block, want, atol=1e-6), block
    assert torch.allclose(r[:4, 4:], torch.zeros(4, 4), atol=1e-6), "跨 group 不应有耦合"
    assert torch.allclose(
        y, torch.tensor([[-0.1997, 0.3841, 1.9900, 3.1389, 2.0482, 1.8794, 6.9141, 8.3945]]), atol=1e-4
    ), y


check_example.example = True


def check_y_equals_matrix_apply(solve):
    """逐元素实现必须等价于 R @ x（三种 variant、两种 batch 形状）。"""
    gen = torch.Generator().manual_seed(3)
    x = torch.randn(2, 3, 8, generator=gen)
    cases = [
        (torch.rand(4, generator=gen), torch.rand(4, generator=gen), "road_1", 4, 8),
        (torch.rand(8, generator=gen), torch.rand(8, generator=gen), "road_2", 4, 8),
        (torch.rand(16, generator=gen), torch.rand(16, generator=gen), "road_4", 4, 8),
    ]
    for theta, alpha, variant, gs, out in cases:
        r, y = solve(theta, alpha, variant, gs, out, x)
        assert torch.allclose(y, x @ r.T, atol=1e-4), (variant, (y - x @ r.T).abs().max())


def check_road1_is_orthogonal(solve):
    """road_1 + alpha=1：每对是纯旋转，R 正交且保范数。"""
    gen = torch.Generator().manual_seed(5)
    theta = torch.rand(8, generator=gen) * 6.28
    x = torch.randn(4, 16, generator=gen)
    r, y = solve(theta, torch.ones(8), "road_1", 4, 16, x)
    assert torch.allclose(r @ r.T, torch.eye(16), atol=1e-5), "两列共用一个角时 R 仍应正交"
    assert torch.allclose(y.norm(dim=1), x.norm(dim=1), atol=1e-4), "旋转应保 2-范数"


def check_road2_alpha_breaks_rotation(solve):
    """road_2 每个位置的 alpha 不同，R 不再正交（只有 alpha 恒等时才是旋转）。"""
    theta = torch.zeros(4)
    alpha = torch.tensor([1.0, 1.0, 2.0, 2.0])
    r, _ = solve(theta, alpha, "road_2", 4, 4, torch.zeros(1, 4))
    assert torch.allclose(r, torch.diag(alpha), atol=1e-6), r
    assert not torch.allclose(r @ r.T, torch.eye(4), atol=1e-5), "alpha 不同时不该再正交"


def check_group_pairing(solve):
    """配对是 group 内前半与后半，不是相邻元素。"""
    theta = torch.tensor([0.5, 0.7])
    r4, _ = solve(theta, torch.ones(2), "road_1", 4, 4, torch.zeros(1, 4))
    assert abs(float(r4[0, 2])) > 0.4 and abs(float(r4[0, 1])) < 1e-6, r4
    theta2 = torch.tensor([0.5, 0.7])
    r2, _ = solve(theta2, torch.ones(2), "road_1", 2, 4, torch.zeros(1, 4))
    assert abs(float(r2[0, 1])) > 0.4, "group_size=2 时配对退化成相邻两维"


def check_road4_two_angles(solve):
    """road_4 的 cos 项与 sin 项来自两段不同的角。"""
    theta = torch.zeros(16)
    theta[0] = 0.0
    theta[4] = 1.2
    alpha = torch.ones(16)
    r, _ = solve(theta, alpha, "road_4", 4, 8, torch.zeros(1, 8))
    assert abs(float(r[0, 0]) - 1.0) < 1e-6, "第一列用前半段的角"
    assert abs(float(r[0, 2]) + math.sin(1.2)) < 1e-6, "第二列用后半段的角"


def check_bad_input(solve):
    """奇数分组、不整除、参数长度不对、x 末维不符都要报错。"""
    x = torch.zeros(1, 8)
    for bad in (
        lambda: solve(torch.zeros(4), torch.ones(4), "road_1", 3, 8, x),
        lambda: solve(torch.zeros(6), torch.ones(6), "road_1", 4, 9, x),
        lambda: solve(torch.zeros(3), torch.ones(3), "road_1", 4, 8, x),
        lambda: solve(torch.zeros(8), torch.ones(8), "road_9", 4, 8, x),
        lambda: solve(torch.zeros(4), torch.ones(4), "road_1", 4, 8, torch.zeros(1, 7)),
        lambda: solve(torch.zeros(4, 1), torch.ones(4), "road_1", 4, 8, x),
    ):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("应当抛出 ValueError")


def check_random_against_oracle(solve):
    """随机 30 组，三种 variant 与矩阵乘法口径对拍。"""
    rng = random.Random(2026)
    gen = torch.Generator().manual_seed(2026)
    for _ in range(30):
        out = rng.choice([4, 8, 12, 16])
        gs = rng.choice([2, 4, 8])
        if out % gs:
            continue
        variant = rng.choice(["road_1", "road_2", "road_4"])
        n = {"road_1": out // 2, "road_2": out, "road_4": 2 * out}[variant]
        theta = torch.rand(n, generator=gen) * 6.28
        alpha = torch.rand(n, generator=gen) * 2
        x = torch.randn(rng.randint(1, 3), out, generator=gen)
        r, y = solve(theta, alpha, variant, gs, out, x)
        want_r, want_y = _oracle(theta, alpha, variant, gs, out, x)
        assert torch.allclose(r, want_r, atol=1e-5), (variant, gs, out, r, want_r)
        assert torch.allclose(y, want_y, atol=1e-4), (variant, gs, out)
