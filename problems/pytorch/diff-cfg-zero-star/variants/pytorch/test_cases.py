import math

import torch


def _alpha(cond_row, uncond_row, eps):
    dot = sum(a * b for a, b in zip(cond_row, uncond_row))
    squared = sum(b * b for b in uncond_row) + eps
    return dot / squared


def _oracle(cond, uncond, scale, step, warmup, eps):
    if step < warmup:
        return torch.zeros(cond.shape, dtype=torch.float64)
    rows = []
    for i in range(cond.shape[0]):
        c = [float(v) for v in cond[i].reshape(-1).tolist()]
        u = [float(v) for v in uncond[i].reshape(-1).tolist()]
        a = _alpha(c, u, eps)
        rows.append(
            torch.tensor([a * ui + scale * (ci - a * ui) for ci, ui in zip(c, u)], dtype=torch.float64)
            .reshape(cond[i].shape)
        )
    return torch.stack(rows)


def check_example(solve):
    c = torch.tensor([[0.4, -1.2, 0.6], [0.0, 0.5, -0.5]])
    u = torch.tensor([[0.1, -0.4, 0.2], [0.2, -0.1, 0.0]])

    got = solve(c, u, 3.0, step=3)
    want = [[0.590476, -1.161905, 0.580952], [0.4, 1.3, -1.5]]
    assert [[round(float(v), 6) for v in row] for row in got.tolist()] == want, got.tolist()
    assert got.dtype == torch.float32 and got.shape == c.shape

    zero = solve(c, u, 3.0, step=0)
    assert torch.equal(zero, torch.zeros_like(c)), "第 0 步落在零初始化窗口内，应输出全零"


check_example.example = True


def check_zero_init_window(solve):
    torch.manual_seed(0)
    c = torch.randn(3, 4, dtype=torch.float64)
    u = torch.randn(3, 4, dtype=torch.float64)

    guided = solve(c, u, 5.0, step=4, zero_init_steps=4)
    assert torch.allclose(guided, _oracle(c, u, 5.0, 4, 0, 1e-8).to(torch.float32), atol=1e-6)
    assert not torch.allclose(guided, torch.zeros_like(guided)), "窗口之后必须恢复引导"

    for step in range(4):
        assert torch.equal(solve(c, u, 5.0, step=step, zero_init_steps=4), torch.zeros_like(c)), step

    # 关闭零初始化后第 0 步就走引导
    active = solve(c, u, 5.0, step=0, zero_init_steps=0)
    assert torch.allclose(active, _oracle(c, u, 5.0, 0, 0, 1e-8).to(torch.float32), atol=1e-6)


def check_scale_one_identity(solve):
    torch.manual_seed(1)
    c = torch.randn(4, 3, 2, dtype=torch.float64)
    u = torch.randn(4, 3, 2, dtype=torch.float64)
    got = solve(c, u, 1.0, step=2)
    assert torch.allclose(got, c.to(torch.float32), atol=1e-7), "s=1 时输出必须等于条件预测"


def check_least_squares_scale(solve):
    torch.manual_seed(2)
    c = torch.randn(4, 5, dtype=torch.float64)
    u = torch.randn(4, 5, dtype=torch.float64) * 1.5
    s = 2.5

    got = solve(c, u, s, step=1).to(torch.float64)
    # pred = s·c + (1-s)·alpha·u，据此反解 alpha·u 并检查残差与 u 正交
    alpha_u = (s * c - got) / (s - 1.0)
    residual = c - alpha_u
    assert torch.allclose((residual * u).sum(dim=1), torch.zeros(4, dtype=torch.float64), atol=1e-6), (
        "alpha 应是最小二乘最优尺度，使残差与 uncond 正交"
    )


def check_degenerate_directions(solve):
    c = torch.tensor([[0.0, 2.0, -1.0], [3.0, -1.0, 0.5]])
    u = torch.tensor([[1.0, 0.0, 0.0], [6.0, -2.0, 1.0]])

    # 两支正交：alpha = 0，结果退化成 s·eps_cond
    got = solve(c, u, 2.0, step=1)
    assert torch.allclose(got[0], 2.0 * c[0], atol=1e-6), "正交时 alpha=0"

    # 两支平行（u = c）：alpha ≈ 1，结果回到条件预测
    parallel = torch.stack([u[1], 2.0 * torch.tensor([3.0, -1.0, 0.5])])
    got = solve(parallel, u[1].unsqueeze(0).repeat(2, 1), 3.0, step=1)
    assert torch.allclose(got[0], parallel[0], atol=1e-6), "平行时 alpha≈1，输出≈eps_cond"


def check_eps_shrinks_scale(solve):
    torch.manual_seed(3)
    c = torch.randn(3, 4, dtype=torch.float64)
    u = torch.randn(3, 4, dtype=torch.float64)
    s = 2.0

    def recovered(eps):
        got = solve(c, u, s, step=1, eps=eps).to(torch.float64)
        alpha_u = (s * c - got) / (s - 1.0)
        return (alpha_u * u).sum(dim=1) / (u * u).sum(dim=1)

    small, large = recovered(1e-8), recovered(2.0)
    assert bool((large.abs() < small.abs()).all()), "eps 增大时最小二乘尺度的绝对值必须变小"


def check_against_oracle(solve):
    torch.manual_seed(4)
    c = torch.randn(3, 4, 2, dtype=torch.float64)
    u = torch.randn(3, 4, 2, dtype=torch.float64) * 0.8

    for s, step, warmup, eps in ((2.0, 1, 1, 1e-8), (7.5, 6, 1, 1e-8), (0.0, 2, 2, 1e-3), (3.0, 0, 1, 5e-2)):
        got = solve(c, u, s, step=step, zero_init_steps=warmup, eps=eps)
        want = _oracle(c, u, s, step, warmup, eps)
        assert torch.allclose(got, want.to(torch.float32), atol=1e-5), (s, step, warmup, eps)


def check_invalid(solve):
    good_c = torch.zeros(2, 3)
    good_u = torch.ones(2, 3)
    for bad in (
        lambda: solve([[0.0, 1.0]], good_u, 2.0, step=1),
        lambda: solve(good_c, torch.zeros(3, 3), 2.0, step=1),
        lambda: solve(torch.zeros(4), torch.ones(4), 2.0, step=1),
        lambda: solve(good_c, good_u, -2.0, step=1),
        lambda: solve(good_c, good_u, 2.0, step=-1),
        lambda: solve(good_c, good_u, 2.0, step=1.5),
        lambda: solve(good_c, good_u, 2.0, step=True),
        lambda: solve(good_c, good_u, 2.0, step=1, zero_init_steps=-1),
        lambda: solve(good_c, good_u, 2.0, step=1, eps=0.0),
        lambda: solve(good_c, good_u, 2.0, step=1, eps=-1e-3),
    ):
        try:
            bad()
            raise AssertionError("非法输入应抛 ValueError")
        except ValueError:
            pass
