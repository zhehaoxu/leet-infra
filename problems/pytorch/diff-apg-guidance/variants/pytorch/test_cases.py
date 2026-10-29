import math

import torch


def _dot(x, y):
    return sum(a * b for a, b in zip(x, y))


def _oracle(cond, uncond, s, eta, tau, mom=None, prev=None, dims=None):
    """逐样本、逐归约切片走 python 标量循环，不复用张量归约。"""
    nd = cond.dim()
    red = tuple(range(1, nd)) if dims is None else tuple(dims)
    keep = tuple(d for d in range(nd) if d not in red)
    perm = keep + red
    width = 1
    for d in red:
        width *= cond.shape[d]

    permuted = cond.permute(perm).to(torch.float64)
    shape = permuted.shape
    cp = permuted.reshape(-1, width)
    up = uncond.permute(perm).to(torch.float64).reshape(-1, width)
    pp = None
    if prev is not None:
        pp = prev.permute(perm).to(torch.float64).reshape(-1, width)

    preds, bufs = [], []
    for i in range(cp.shape[0]):
        c = [float(v) for v in cp[i].tolist()]
        u = [float(v) for v in up[i].tolist()]
        diff = [x - y for x, y in zip(c, u)]
        if mom is not None:
            base = [0.0] * width if pp is None else [float(v) for v in pp[i].tolist()]
            diff = [x + mom * y for x, y in zip(diff, base)]
            bufs.append(diff)
        if tau > 0.0:
            norm = math.sqrt(_dot(diff, diff))
            factor = min(1.0, tau / norm) if norm > 0.0 else 1.0
            diff = [x * factor for x in diff]
        cnorm = math.sqrt(_dot(c, c))
        unit = [x / max(cnorm, 1e-12) for x in c]
        projected = _dot(diff, unit)
        update = [d + (eta - 1.0) * projected * e for d, e in zip(diff, unit)]
        preds.append([y + s * x for x, y in zip(update, u)])

    inv = torch.argsort(torch.tensor(perm)).tolist()
    out = torch.tensor(preds, dtype=torch.float64).reshape(shape).permute(inv).contiguous()
    buf = None
    if mom is not None:
        buf = torch.tensor(bufs, dtype=torch.float64).reshape(shape).permute(inv).contiguous()
    return out, buf


def check_example(solve):
    c = torch.tensor([[0.4, -1.2, 0.6], [0.0, 0.5, -0.5]])
    u = torch.tensor([[0.1, -0.4, 0.2], [0.2, -0.1, 0.0]])

    pred, buf = solve(c, u, 3.0, eta=0.5)
    want = [[0.595918, -1.587755, 0.793878], [-0.4, 0.875, -0.675]]
    assert [[round(float(v), 6) for v in row] for row in pred.tolist()] == want, pred.tolist()
    assert buf is None, "未启用动量时第二项必须是 None"
    assert pred.dtype == torch.float32 and pred.shape == c.shape

    plain, _ = solve(c, u, 3.0)
    assert [[round(float(v), 6) for v in row] for row in plain.tolist()] == [
        [1.0, -2.8, 1.4],
        [-0.4, 1.7, -1.5],
    ], "eta=1 必须退化成裸 CFG"


check_example.example = True


def check_eta_one_is_plain_cfg(solve):
    torch.manual_seed(0)
    c = torch.randn(3, 4, 5, dtype=torch.float64)
    u = torch.randn(3, 4, 5, dtype=torch.float64) * 0.5
    for s in (0.0, 2.0, 7.5):
        got, _ = solve(c, u, s, eta=1.0)
        want = (u + s * (c - u)).to(torch.float32)
        assert torch.allclose(got, want, atol=1e-6), f"eta=1、s={s} 应等于裸 CFG"


def check_eta_zero_drops_parallel(solve):
    torch.manual_seed(1)
    c = torch.randn(4, 6, dtype=torch.float64)
    u = torch.randn(4, 6, dtype=torch.float64)
    s = 2.5

    got, _ = solve(c, u, s, eta=0.0)
    update = (got.to(torch.float64) - u) / s
    # 更新向量与条件预测正交：平行分量被完全丢弃
    assert torch.allclose((update * c).sum(dim=1), torch.zeros(4, dtype=torch.float64), atol=1e-6)

    # eta 在「全正交」与「全保留」两端之间线性收缩
    zero = (got.to(torch.float64) - u) / s
    full = (solve(c, u, s, eta=1.0)[0].to(torch.float64) - u) / s
    half = (solve(c, u, s, eta=0.5)[0].to(torch.float64) - u) / s
    assert torch.allclose(half, 0.5 * zero + 0.5 * full, atol=1e-6), "eta 必须线性插值两个端点"


def check_norm_threshold(solve):
    c = torch.tensor([[10.0, 0.0], [0.3, 0.0]])
    u = torch.tensor([[6.0, 0.0], [0.1, 0.0]])

    got, _ = solve(c, u, 2.0, norm_threshold=1.0)
    assert torch.allclose(got, torch.tensor([[8.0, 0.0], [0.5, 0.0]]), atol=1e-6), got

    # 差分范数没超阈值的那一例必须与不截断时逐位一致
    free, _ = solve(c, u, 2.0)
    assert torch.allclose(got[1], free[1], atol=1e-7), "范数未超阈值时不应截断"

    # 截断是逐样本的：更新向量范数不超过阈值
    torch.manual_seed(2)
    big_c = torch.randn(5, 3, 2, dtype=torch.float64) * 10.0
    big_u = torch.randn(5, 3, 2, dtype=torch.float64)
    for tau in (0.5, 3.0):
        clamped, _ = solve(big_c, big_u, 1.7, eta=0.9, norm_threshold=tau)
        upd = (clamped.to(torch.float64) - big_u) / 1.7
        norms = upd.flatten(1).norm(dim=1)
        assert bool((norms <= tau + 1e-5).all()), f"更新范数超过阈值 tau={tau}: {norms.tolist()}"

    # 两条支路完全相同时差分为零，不能被 0/0 污染成 NaN
    same, _ = solve(torch.ones(2, 3), torch.ones(2, 3), 4.0, norm_threshold=0.5)
    assert torch.isfinite(same).all() and torch.allclose(same, torch.ones(2, 3), atol=1e-6)


def check_momentum_state(solve):
    torch.manual_seed(3)
    c = torch.randn(3, 4, dtype=torch.float64)
    u = torch.randn(3, 4, dtype=torch.float64)
    m = 0.6

    _, buf0 = solve(c, u, 3.0, momentum=m)
    assert torch.allclose(buf0, (c - u).to(torch.float32), atol=1e-6), "首步缓冲就是原始差分"

    nxt = c * 0.5 + 0.25
    pred1, buf1 = solve(nxt, u, 3.0, momentum=m, prev_momentum=buf0)
    manual = (nxt - u) + m * buf0.to(torch.float64)
    assert torch.allclose(buf1, manual.to(torch.float32), atol=1e-6), "缓冲递推应为 v + m·prev"

    again, _ = solve(nxt, u, 3.0, momentum=m, prev_momentum=buf0)
    assert torch.allclose(pred1, again, atol=1e-7), "同状态重算必须逐位一致"

    # 截断只作用于本步更新，不写回动量缓冲
    _, buf2 = solve(torch.full((1, 3), 50.0), torch.zeros(1, 3), 2.0, norm_threshold=1.0, momentum=0.9)
    assert torch.allclose(buf2, torch.full((1, 3), 50.0), atol=1e-6), "缓冲必须保存截断前的差分"


def check_against_oracle(solve):
    torch.manual_seed(4)
    c = torch.randn(4, 3, 2, dtype=torch.float64)
    u = torch.randn(4, 3, 2, dtype=torch.float64) * 0.7

    configs = [
        dict(s=3.0, eta=1.0, tau=0.0),
        dict(s=3.0, eta=0.3, tau=0.0),
        dict(s=2.0, eta=0.5, tau=1.0),
        dict(s=0.0, eta=0.0, tau=0.4),
    ]
    for cfg in configs:
        got, _ = solve(c, u, cfg["s"], eta=cfg["eta"], norm_threshold=cfg["tau"])
        want, _ = _oracle(c, u, cfg["s"], cfg["eta"], cfg["tau"])
        assert torch.allclose(got, want.to(torch.float32), atol=1e-5), cfg

    prev = torch.randn(4, 3, 2, dtype=torch.float64) * 0.2
    got, buf = solve(c, u, 2.0, eta=0.4, norm_threshold=0.8, momentum=0.5, prev_momentum=prev)
    want, want_buf = _oracle(c, u, 2.0, 0.4, 0.8, mom=0.5, prev=prev)
    assert torch.allclose(got, want.to(torch.float32), atol=1e-5)
    assert torch.allclose(buf, want_buf.to(torch.float32), atol=1e-5)

    # 显式 dims：只在最后一维上做投影
    got, _ = solve(c, u, 2.0, eta=0.25, norm_threshold=0.6, dims=(2,))
    want, _ = _oracle(c, u, 2.0, 0.25, 0.6, dims=(2,))
    assert torch.allclose(got, want.to(torch.float32), atol=1e-5), "dims=(2,) 与 oracle 不一致"


def check_invalid(solve):
    good_c = torch.zeros(2, 3)
    good_u = torch.ones(2, 3)
    for bad in (
        lambda: solve([[0.0, 1.0]], good_u, 2.0),
        lambda: solve(good_c, torch.zeros(3, 3), 2.0),
        lambda: solve(torch.zeros(4), torch.ones(4), 2.0),
        lambda: solve(good_c, good_u, -1.0),
        lambda: solve(good_c, good_u, 2.0, eta=1.5),
        lambda: solve(good_c, good_u, 2.0, eta=-0.1),
        lambda: solve(good_c, good_u, 2.0, norm_threshold=-1.0),
        lambda: solve(good_c, good_u, 2.0, momentum=1.0),
        lambda: solve(good_c, good_u, 2.0, momentum=-0.2),
        lambda: solve(good_c, good_u, 2.0, prev_momentum=good_c),
        lambda: solve(good_c, good_u, 2.0, momentum=0.5, prev_momentum=torch.zeros(3, 2)),
        lambda: solve(good_c, good_u, 2.0, dims=(0,)),
        lambda: solve(good_c, good_u, 2.0, dims=(1, 1)),
        lambda: solve(good_c, good_u, 2.0, dims=(2,)),
        lambda: solve(good_c, good_u, 2.0, dims=()),
    ):
        try:
            bad()
            raise AssertionError("非法输入应抛 ValueError")
        except ValueError:
            pass
