"""独立 oracle：用 Python 循环逐元素算行/列范数与 delta，不借助 torch 的范数 API。"""

import math
import random

import torch


def _oracle(W0, A, B, lam, w_norm):
    r = len(A)
    n_in = len(A[0])
    n_out = len(B)
    an = [max(math.sqrt(sum(v * v for v in A[i])), 1e-4) for i in range(r)]
    bn = [max(math.sqrt(sum(B[i][j] * B[i][j] for i in range(n_out))), 1e-4) for j in range(r)]
    scaling = [lam / r / (an[i] * bn[i]) for i in range(r)]
    delta = [
        [sum(B[i][k] * scaling[k] * A[k][j] for k in range(r)) * w_norm[j] for j in range(n_in)]
        for i in range(n_out)
    ]
    merged = [[W0[i][j] + delta[i][j] for j in range(n_in)] for i in range(n_out)]
    return scaling, delta, merged


def check_example(solve):
    """缩放只看 A 的行范数与 B 的列范数；w_norm 按输入通道缩放 delta 的列。"""
    w0 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    a = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    b = torch.tensor([[1.0, 0.0], [0.0, 3.0]])
    w_norm = torch.tensor([1.0, 2.0, 1.0])
    scaling, delta, merged = solve(w0, a, b, 1.5, w_norm)
    assert torch.allclose(scaling, torch.tensor([0.75, 0.125]), atol=1e-6), scaling
    assert torch.allclose(delta, torch.tensor([[0.75, 0.0, 0.0], [0.0, 1.5, 0.0]]), atol=1e-6), delta
    assert torch.allclose(merged, torch.tensor([[1.75, 2.0, 3.0], [4.0, 6.5, 6.0]]), atol=1e-6), merged


check_example.example = True


def check_norm_bound(solve):
    """A 的行与 B 的列都单位化时，delta 的 Frobenius 范数恰好是 lam/sqrt(r)。"""
    w0 = torch.zeros(3, 3)
    a = torch.eye(3)[:2]
    b = torch.eye(3)[:, :2]
    _, delta, _ = solve(w0, a, b, 2.0, torch.ones(3))
    want = 2.0 / math.sqrt(2.0)
    assert abs(float(delta.norm()) - want) < 1e-5, (float(delta.norm()), want)
    _, delta1, _ = solve(w0, a[:1], b[:, :1], 2.0, torch.ones(3))
    assert abs(float(delta1.norm()) - 2.0) < 1e-5, "r=1 时范数就等于 lam"


def check_frobenius_clamp(solve):
    """零行/零列不能把缩放推到无穷：范数下限 1e-4。"""
    w0 = torch.zeros(2, 3)
    a = torch.tensor([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    b = torch.tensor([[1.0, 0.0], [0.0, 3.0]])
    scaling, delta, _ = solve(w0, a, b, 1.0, torch.ones(3))
    assert torch.allclose(scaling, torch.tensor([5000.0, 1.0 / 12.0]), rtol=1e-5), scaling
    assert torch.isfinite(delta).all()


def check_w_norm_scales_columns(solve):
    """w_norm 只按列（输入通道）缩放，不改变行方向。"""
    gen = torch.Generator().manual_seed(21)
    a = torch.randn(2, 4, generator=gen)
    b = torch.randn(3, 2, generator=gen)
    w0 = torch.zeros(3, 4)
    one = solve(w0, a, b, 0.8, torch.ones(4))[1]
    scaled = solve(w0, a, b, 0.8, torch.tensor([1.0, 2.0, 0.5, 3.0]))[1]
    assert torch.allclose(scaled, one * torch.tensor([[1.0, 2.0, 0.5, 3.0]]), atol=1e-5), scaled


def check_merge_matches_forward(solve):
    """合并权重与 DeLoRA 的三段式前向（乘 w_norm → A → 缩放 → B）等价。"""
    gen = torch.Generator().manual_seed(33)
    w0 = torch.randn(3, 4, generator=gen)
    a = torch.randn(2, 4, generator=gen)
    b = torch.randn(3, 2, generator=gen)
    w_norm = torch.rand(4, generator=gen) + 0.5
    x = torch.randn(5, 4, generator=gen)
    scaling, _, merged = solve(w0, a, b, 1.2, w_norm)
    h = (x * w_norm) @ a.T * scaling
    want = x @ w0.T + h @ b.T
    assert torch.allclose(x @ merged.T, want, atol=1e-4), (x @ merged.T - want).abs().max()


def check_rank_comes_from_A(solve):
    """rank 由 A 的行数决定；B 的列数对不上要报错。"""
    a = torch.randn(3, 2)
    b = torch.randn(4, 3)
    w0 = torch.zeros(4, 2)
    scaling, _, _ = solve(w0, a, b, 1.0, torch.ones(2))
    assert scaling.shape == (3,), scaling.shape
    try:
        solve(w0, a, torch.randn(4, 2), 1.0, torch.ones(2))
    except ValueError:
        pass
    else:
        raise AssertionError("rank 不一致应当报错")


def check_bad_input(solve):
    """维度、形状、负的 lam 都要报错。"""
    a = torch.randn(2, 3)
    b = torch.randn(2, 2)
    w0 = torch.zeros(2, 3)
    for bad in (
        lambda: solve(w0, torch.randn(2, 3, 1), b, 1.0, torch.ones(3)),
        lambda: solve(torch.zeros(3, 3), a, b, 1.0, torch.ones(3)),
        lambda: solve(w0, a, b, 1.0, torch.ones(2)),
        lambda: solve(w0, a, b, -0.5, torch.ones(3)),
        lambda: solve(w0, a, b, 1.0, torch.ones(3, 1)),
    ):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("应当抛出 ValueError")


def check_random_against_oracle(solve):
    """随机 30 组与纯 Python oracle 对拍。"""
    rng = random.Random(9204)
    gen = torch.Generator().manual_seed(9204)
    for _ in range(30):
        r = rng.randint(1, 3)
        n_in = rng.randint(1, 4)
        n_out = rng.randint(1, 4)
        a = torch.randn(r, n_in, generator=gen)
        b = torch.randn(n_out, r, generator=gen)
        w0 = torch.randn(n_out, n_in, generator=gen)
        w_norm = torch.rand(n_in, generator=gen) + 0.2
        lam = rng.uniform(0.0, 3.0)
        scaling, delta, merged = solve(w0, a, b, lam, w_norm)
        want_s, want_d, want_m = _oracle(
            w0.tolist(), a.tolist(), b.tolist(), lam, w_norm.tolist()
        )
        assert torch.allclose(scaling, torch.tensor(want_s), rtol=1e-4, atol=1e-6), (scaling, want_s)
        assert torch.allclose(delta, torch.tensor(want_d), rtol=1e-4, atol=1e-6), (delta, want_d)
        assert torch.allclose(merged, torch.tensor(want_m), rtol=1e-4, atol=1e-6)
