import math

import torch


def _oracle(cond, uncond, scale):
    """照 diffusers 的写法：对 [cond; uncond] 做 SVD，把 uncond 投到第一右奇异向量上。"""
    c = cond.to(torch.float64)
    u = uncond.to(torch.float64)
    stacked = torch.stack([c, u], dim=1).flatten(2)
    _, _, vh = torch.linalg.svd(stacked, full_matrices=False)
    truncated = vh.clone()
    truncated[:, 1] = 0.0
    flat = u.reshape(u.size(0), 1, -1)
    projected = torch.matmul(torch.matmul(flat, vh.transpose(-2, -1)), truncated).reshape(u.shape)
    return projected + scale * (c - projected)


def check_example(solve):
    c = torch.tensor([[0.4, -1.2, 0.6], [0.0, 0.5, -0.5]])
    u = torch.tensor([[0.1, -0.4, 0.2], [0.2, -0.1, 0.0]])

    got = solve(c, u, 3.0)
    want = [[0.944568, -2.814419, 1.407210], [-0.004763, 1.610847, -1.608465]]
    assert [[round(float(v), 6) for v in row] for row in got.tolist()] == want, got.tolist()
    assert got.dtype == torch.float32 and got.shape == c.shape

    proj = solve(c, u, 0.0)
    want_proj = [[0.127716, -0.392791, 0.196395], [0.002381, -0.055423, 0.054233]]
    assert [[round(float(v), 6) for v in row] for row in proj.tolist()] == want_proj, proj.tolist()


check_example.example = True


def check_scale_one_identity(solve):
    torch.manual_seed(0)
    c = torch.randn(4, 6, dtype=torch.float64)
    u = torch.randn(4, 6, dtype=torch.float64)
    got = solve(c, u, 1.0)
    assert torch.equal(got, c.to(torch.float32)), "s=1 时输出必须逐位等于条件预测"


def check_projection_geometry(solve):
    torch.manual_seed(1)
    c = torch.randn(5, 3, 4, dtype=torch.float64)
    u = torch.randn(5, 3, 4, dtype=torch.float64) * 0.6

    proj = solve(c, u, 0.0).to(torch.float64)
    # 投影点与残差正交：proj 落在第一右奇异向量上，u - proj 与之垂直
    dot = (proj * (u - proj)).flatten(1).sum(dim=1)
    assert torch.allclose(dot, torch.zeros(5, dtype=torch.float64), atol=1e-6), "投影残差应与投影方向正交"
    # 逐样本只做一次标量缩放：proj 与 u 平行分量之外的部分被压到 0
    assert not torch.allclose(proj, u, atol=1e-3), "投影必须真的改变 uncond，否则等于没做"

    # s 只是沿 cond - proj 方向的外推，落点始终在 proj 这条射线上
    for s in (2.0, 5.5):
        got = solve(c, u, s).to(torch.float64)
        assert torch.allclose(got - proj, s * (c - proj), atol=1e-5), f"s={s} 的外推方向应一致"


def check_tie_and_degenerate(solve):
    # cond ⟂ uncond 且模长相同：两个奇异值相等，主方向按约定取 cond 方向，投影为 0
    tie_c = torch.tensor([[1.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    tie_u = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 3.0]])
    got = solve(tie_c, tie_u, 2.0)
    assert torch.allclose(got, 2.0 * tie_c, atol=1e-6), "奇异值并列时应取 cond 方向，投影退化为 0"

    # 两支平行：主方向就是公共方向，uncond 完整保留，等价于裸 CFG
    c = torch.tensor([[1.0, 2.0, -1.0]])
    u = 0.5 * c
    got = solve(c, u, 3.0)
    want = u + 3.0 * (c - u)
    assert torch.allclose(got, want, atol=1e-6), "平行输入必须退化成裸 CFG"

    # uncond 全零：投影为 0，输出 s·cond
    zero_u = torch.zeros(1, 3)
    got = solve(c, zero_u, 2.5)
    assert torch.allclose(got, 2.5 * c, atol=1e-6)
    assert torch.equal(solve(torch.zeros(1, 3), torch.zeros(1, 3), 4.0), torch.zeros(1, 3))


def check_against_oracle(solve):
    torch.manual_seed(2)
    for trial in range(6):
        shape = (4, 3, 2) if trial % 2 else (6, 5)
        c = torch.randn(*shape, dtype=torch.float64) * (1 + trial)
        u = torch.randn(*shape, dtype=torch.float64) * 0.5
        for s in (0.0, 1.5, 4.0):
            got = solve(c, u, s)
            want = _oracle(c, u, s)
            assert torch.allclose(got, want.to(torch.float32), atol=1e-5), (trial, s)


def check_invalid(solve):
    good_c = torch.zeros(2, 3)
    good_u = torch.ones(2, 3)
    for bad in (
        lambda: solve([[0.0, 1.0]], good_u, 2.0),
        lambda: solve(good_c, torch.zeros(3, 3), 2.0),
        lambda: solve(torch.zeros(4), torch.ones(4), 2.0),
        lambda: solve(good_c, good_u, -1.0),
        lambda: solve(good_c, good_u, float("inf")),
        lambda: solve(good_c, good_u, float("nan")),
    ):
        try:
            bad()
            raise AssertionError("非法输入应抛 ValueError")
        except ValueError:
            pass
