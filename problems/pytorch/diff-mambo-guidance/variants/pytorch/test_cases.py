import math

import torch


def _oracle(cond, uncond, scale, alpha):
    """独立 oracle：逐样本用 python 标量求两个 L2 范数与指数衰减。"""
    rows_c = cond.reshape(cond.shape[0], -1).tolist()
    rows_u = uncond.reshape(uncond.shape[0], -1).tolist()
    rows = []
    for rc, ru in zip(rows_c, rows_u):
        diff = [a - b for a, b in zip(rc, ru)]
        norm_diff = math.sqrt(sum(v * v for v in diff))
        norm_uncond = math.sqrt(sum(v * v for v in ru))
        if norm_uncond == 0.0:
            scale_final = 1.0
        else:
            scale_final = 1.0 + (scale - 1.0) * math.exp(-alpha * norm_diff / norm_uncond)
        rows.append([b + scale_final * d for b, d in zip(ru, diff)])
    return torch.tensor(rows, dtype=torch.float64).reshape(cond.shape)


def check_example(solve):
    c = torch.tensor([[1.5, -2.5, 1.25], [-0.25, 1.0, -1.5]])
    u = torch.tensor([[1.0, -2.0, 1.0], [0.5, 1.0, -1.0]])

    got = solve(c, u, 10.0, alpha=0.5)
    assert [[round(float(v), 6) for v in row] for row in got.tolist()] == [
        [5.361224, -6.361224, 3.180612],
        [-5.248210, 1.0, -4.832140],
    ], got.tolist()
    assert got.dtype == torch.float32 and got.shape == c.shape

    raw = solve(c, u, 10.0, alpha=0.0)
    assert [[round(float(v), 6) for v in row] for row in raw.tolist()] == [
        [6.0, -7.0, 3.5],
        [-7.0, 1.0, -6.0],
    ], "alpha=0 时应是裸 CFG"

    strong = solve(c, u, 10.0, alpha=2.0)
    assert [[round(float(v), 6) for v in row] for row in strong.tolist()] == [
        [3.939285, -4.939285, 2.469642],
        [-2.279302, 1.0, -2.852868],
    ], strong.tolist()


check_example.example = True


def check_alpha_zero_is_plain_cfg(solve):
    torch.manual_seed(0)
    c = torch.randn(3, 2, 4, dtype=torch.float64)
    u = torch.randn(3, 2, 4, dtype=torch.float64)

    for scale in (1.0, 3.0, 10.0):
        got = solve(c, u, scale, alpha=0.0).to(torch.float64)
        assert torch.allclose(got, u + scale * (c - u), atol=1e-6), f"alpha=0、s={scale} 应退化为 CFG"


def check_scale_one_identity(solve):
    torch.manual_seed(1)
    c = torch.randn(3, 2, 4, dtype=torch.float64)
    u = torch.randn(3, 2, 4, dtype=torch.float64)

    # w' = 1 + (1-1)·exp(...) = 1，与 alpha、幅值比无关
    for alpha in (0.0, 8.0, 100.0):
        got = solve(c, u, 1.0, alpha=alpha)
        assert torch.equal(got, c.to(torch.float32)), f"scale=1、alpha={alpha} 必须原样返回条件预测"


def check_scale_final_is_per_sample(solve):
    torch.manual_seed(2)
    c = torch.randn(4, 3, 5, dtype=torch.float64)
    u = torch.randn(4, 3, 5, dtype=torch.float64) + 2.0

    got = solve(c, u, 7.5, alpha=0.5).to(torch.float64)
    ratio = (got - u) / (c - u)
    assert torch.allclose(ratio, ratio[:, :1, :1].expand_as(ratio), atol=1e-5), (
        "每个样本只有一个标量缩放因子"
    )
    # 该因子等于 1 + (s-1)·exp(-λ·‖delta‖/‖u‖)
    want = 1.0 + 6.5 * torch.exp(-0.5 * (c - u).flatten(1).norm(dim=1) / u.flatten(1).norm(dim=1))
    assert torch.allclose(ratio[:, 0, 0], want, atol=1e-4), "缩放因子应等于幅值比衰减式"


def check_suppression_monotone(solve):
    torch.manual_seed(3)
    c = torch.randn(3, 4, dtype=torch.float64)
    u = torch.randn(3, 4, dtype=torch.float64)

    distances = []
    for alpha in (0.0, 0.5, 2.0, 8.0, 64.0):
        got = solve(c, u, 10.0, alpha=alpha).to(torch.float64)
        distances.append(float((got - c).norm()))
    assert all(a > b for a, b in zip(distances, distances[1:])), f"alpha 越大应越贴回条件预测: {distances}"
    assert distances[-1] < 1e-6 * max(distances[0], 1.0), "alpha 很大时输出应收敛到条件预测"

    # 幅值比越大，同样 alpha 下缩放因子越贴近 1：构造两个样本，一个 diff 大一个 diff 小
    u2 = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
    c2 = torch.tensor([[4.0, 0.0], [1.1, 0.0]], dtype=torch.float64)
    out = solve(c2, u2, 10.0, alpha=1.0).to(torch.float64)
    scale_final = ((out - u2) / (c2 - u2)).flatten()
    assert 1.0 < scale_final[0] < scale_final[2] < 10.0, (
        f"幅值比大的样本应被压得更接近 1: {scale_final.tolist()}"
    )


def check_zero_uncond(solve):
    c = torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.float64)
    u = torch.zeros_like(c)

    got = solve(c, u, 10.0, alpha=8.0)
    assert torch.equal(got, c.to(torch.float32)), "‖uncond‖=0 时输出应为条件预测"
    assert torch.isfinite(got).all(), "零范数不能产生 NaN / inf"


def check_against_oracle(solve):
    torch.manual_seed(4)
    c = torch.randn(3, 2, 4, dtype=torch.float64)
    u = torch.randn(3, 2, 4, dtype=torch.float64)

    for scale, alpha in ((10.0, 8.0), (7.5, 0.0), (3.0, 0.5), (1.0, 2.0), (12.0, 64.0)):
        got = solve(c, u, scale, alpha)
        want = _oracle(c, u, scale, alpha)
        assert torch.allclose(got, want.to(torch.float32), atol=1e-5), f"s={scale}, alpha={alpha} 与 oracle 不一致"

    # 混合幅值比的样本（含一个近乎零的无条件预测）
    c2 = torch.randn(4, 3, dtype=torch.float64)
    u2 = c2 * 0.1
    u2[1] = 1e-30
    got = solve(c2, u2, 9.0, alpha=4.0)
    want = _oracle(c2, u2, 9.0, 4.0)
    assert torch.allclose(got, want.to(torch.float32), atol=1e-5), "混合幅值比与 oracle 不一致"


def check_invalid(solve):
    c = torch.zeros(2, 3)
    u = torch.ones(2, 3)

    for bad in (
        lambda: solve([[0.0, 1.0]], u, 7.5),
        lambda: solve(c, torch.ones(3, 3), 7.5),
        lambda: solve(torch.zeros(4), torch.ones(4), 7.5),
        lambda: solve(torch.tensor(1.0), torch.tensor(1.0), 7.5),
        lambda: solve(c, u, -1.0),
        lambda: solve(c, u, 7.5, alpha=-0.1),
        lambda: solve(c, u, 7.5, alpha=float("nan")),
        lambda: solve(c, u, float("inf")),
        lambda: solve(c, u, "7.5"),
        lambda: solve(c, u, 7.5, alpha=True),
    ):
        try:
            bad()
            raise AssertionError("非法输入应抛 ValueError")
        except ValueError:
            pass
