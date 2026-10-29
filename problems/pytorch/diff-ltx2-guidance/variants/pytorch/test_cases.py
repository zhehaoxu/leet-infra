import math

import torch


def _oracle(cond, uncond, stg, modality, step, steps, cfg, stg_scale, mod_scale, start, stop):
    """独立 oracle：窗口与三项增量全部用 python 标量重做。"""
    within = int(start * steps) <= step < int(stop * steps)
    cfg_on = within and not math.isclose(cfg, 1.0)
    stg_on = within and not math.isclose(stg_scale, 0.0)
    mod_on = within and not math.isclose(mod_scale, 1.0)

    rows_c = cond.reshape(cond.shape[0], -1).tolist()
    rows_u = uncond.reshape(uncond.shape[0], -1).tolist()
    rows_s = stg.reshape(stg.shape[0], -1).tolist()
    rows_m = modality.reshape(modality.shape[0], -1).tolist()

    out = []
    for rc, ru, rs, rm in zip(rows_c, rows_u, rows_s, rows_m):
        row = []
        for c, u, s, m in zip(rc, ru, rs, rm):
            v = c
            if cfg_on:
                v += (cfg - 1.0) * (c - u)
            if stg_on:
                v += stg_scale * (c - s)
            if mod_on:
                v += (mod_scale - 1.0) * (c - m)
            row.append(v)
        out.append(row)
    return torch.tensor(out, dtype=torch.float64).reshape(cond.shape)


C = torch.tensor([[0.4, -1.2, 0.6], [0.0, 0.5, -0.5]])
U = torch.tensor([[0.1, -0.4, 0.2], [0.2, -0.1, 0.0]])
S = torch.tensor([[0.2, -0.8, 0.5], [0.1, 0.3, -0.2]])
M = torch.tensor([[0.3, -1.0, 0.55], [0.05, 0.2, -0.3]])


def check_example(solve):
    # 默认尺度：三项全部恒等，输出就是条件预测
    plain = solve(C, U, S, M, 1, 10)
    assert torch.equal(plain, C), plain.tolist()
    assert plain.dtype == torch.float32 and plain.shape == C.shape

    full = solve(C, U, S, M, 1, 10, cfg_scale=3.0, stg_scale=1.0, modality_scale=1.5)
    assert [[round(float(v), 6) for v in row] for row in full.tolist()] == [
        [1.25, -3.3, 1.525],
        [-0.525, 2.05, -1.9],
    ], full.tolist()

    cfg_only = solve(C, U, None, None, 1, 10, cfg_scale=3.0)
    assert [[round(float(v), 6) for v in row] for row in cfg_only.tolist()] == [
        [1.0, -2.8, 1.4],
        [-0.4, 1.7, -1.5],
    ], cfg_only.tolist()


check_example.example = True


def check_identity_defaults(solve):
    torch.manual_seed(0)
    c = torch.randn(3, 2, 4, dtype=torch.float64)
    u = torch.randn(3, 2, 4, dtype=torch.float64)
    s = torch.randn(3, 2, 4, dtype=torch.float64)
    m = torch.randn(3, 2, 4, dtype=torch.float64)

    # 三个尺度都在恒等点，三项全关，且可以完全不传三路预测
    assert torch.equal(solve(c, None, None, None, 0, 10), c.to(torch.float32))
    assert torch.equal(solve(c, u, s, m, 0, 10), c.to(torch.float32))
    # 单项恒等：cfg=1.0 只关 CFG，另两项照常
    got = solve(c, None, s, m, 5, 10, cfg_scale=1.0, stg_scale=2.0, modality_scale=1.0)
    assert torch.allclose(got, (c + 2.0 * (c - s)).to(torch.float32), atol=1e-6)


def check_additivity(solve):
    torch.manual_seed(1)
    c = torch.randn(2, 3, 5, dtype=torch.float64)
    u = torch.randn(2, 3, 5, dtype=torch.float64)
    s = torch.randn(2, 3, 5, dtype=torch.float64)
    m = torch.randn(2, 3, 5, dtype=torch.float64)

    cfg, stg, mod = 5.0, 2.5, 1.8
    full = solve(c, u, s, m, 2, 10, cfg, stg, mod).to(torch.float64)
    cfg_only = solve(c, u, None, None, 2, 10, cfg, 0.0, 1.0).to(torch.float64)
    stg_only = solve(c, None, s, None, 2, 10, 1.0, stg, 1.0).to(torch.float64)
    mod_only = solve(c, None, None, m, 2, 10, 1.0, 0.0, mod).to(torch.float64)

    cd = c.to(torch.float64)
    assert torch.allclose(full - cd, (cfg - 1.0) * (cd - u) + stg * (cd - s) + (mod - 1.0) * (cd - m), atol=1e-5)
    assert torch.allclose(full, cfg_only + stg_only + mod_only - 2 * cd, atol=1e-5), "三项应可加"
    # 每一项单独都从 cond 出发，而不是在上一项结果上叠加
    assert torch.allclose(cfg_only - cd, (cfg - 1.0) * (cd - u), atol=1e-5)
    assert torch.allclose(stg_only - cd, stg * (cd - s), atol=1e-5)
    assert torch.allclose(mod_only - cd, (mod - 1.0) * (cd - m), atol=1e-5)


def check_scale_semantics(solve):
    torch.manual_seed(2)
    c = torch.randn(2, 4, dtype=torch.float64)
    u = torch.randn(2, 4, dtype=torch.float64)
    s = torch.randn(2, 4, dtype=torch.float64)
    m = torch.randn(2, 4, dtype=torch.float64)

    # cfg / modality 是「以 1 为恒等」：尺度 2 等于加一份差值
    two = solve(c, u, None, None, 1, 10, cfg_scale=2.0).to(torch.float64)
    assert torch.allclose(two, c + (c - u), atol=1e-6)
    mod_two = solve(c, None, None, m, 1, 10, modality_scale=2.0).to(torch.float64)
    assert torch.allclose(mod_two, c + (c - m), atol=1e-6)
    # stg 是「以 0 为恒等」：尺度 1 就是加一份差值
    stg_one = solve(c, None, s, None, 1, 10, stg_scale=1.0).to(torch.float64)
    assert torch.allclose(stg_one, c + (c - s), atol=1e-6)

    # 尺度退化判定：cfg=1.0 / stg=0.0 / modality=1.0 各自关闭，而不是把差值乘零
    off = solve(c, u, s, m, 1, 10, cfg_scale=1.0, stg_scale=0.0, modality_scale=1.0)
    assert torch.equal(off, c.to(torch.float32))
    # cfg 用 isclose 判退化：1.0 + 1e-12 仍算关闭
    almost = solve(c, u, None, None, 1, 10, cfg_scale=1.0 + 1e-12)
    assert torch.equal(almost, c.to(torch.float32))


def check_window(solve):
    torch.manual_seed(3)
    c = torch.randn(2, 4, dtype=torch.float64)
    u = torch.randn(2, 4, dtype=torch.float64)
    s = torch.randn(2, 4, dtype=torch.float64)
    m = torch.randn(2, 4, dtype=torch.float64)

    # 三路共用同一窗口 [3, 7)：左端点含、右端点不含
    kw = dict(start=0.3, stop=0.7, cfg_scale=3.0, stg_scale=1.0, modality_scale=1.5)
    assert torch.equal(solve(c, u, s, m, 2, 10, **kw), c.to(torch.float32)), "窗口左侧外"
    assert torch.equal(solve(c, u, s, m, 7, 10, **kw), c.to(torch.float32)), "窗口右侧外"
    inside = solve(c, u, s, m, 3, 10, **kw)
    assert not torch.allclose(inside, c.to(torch.float32), atol=1e-4), "左端点应生效"

    # 窗口外即使三个尺度都在非恒等点，也不需要任何分支预测
    assert torch.equal(solve(c, None, None, None, 9, 10, **kw), c.to(torch.float32))


def check_against_oracle(solve):
    torch.manual_seed(4)
    c = torch.randn(3, 2, 4, dtype=torch.float64)
    u = torch.randn(3, 2, 4, dtype=torch.float64)
    s = torch.randn(3, 2, 4, dtype=torch.float64)
    m = torch.randn(3, 2, 4, dtype=torch.float64)

    settings = [
        (1.0, 0.0, 1.0, 0.0, 1.0),
        (7.5, 2.8, 1.0, 0.0, 1.0),
        (3.0, 1.0, 1.5, 0.0, 1.0),
        (4.0, 2.0, 1.2, 0.2, 0.85),
        (2.0, 0.5, 1.9, 0.5, 0.5),
    ]
    for cfg, stg, mod, start, stop in settings:
        for step in range(10):
            got = solve(c, u, s, m, step, 10, cfg, stg, mod, start, stop)
            want = _oracle(c, u, s, m, step, 10, cfg, stg, mod, start, stop)
            assert torch.allclose(got, want.to(torch.float32), atol=1e-5), (
                f"cfg={cfg}, stg={stg}, mod={mod}, step={step} 与 oracle 不一致"
            )


def check_invalid(solve):
    c = torch.zeros(2, 3)
    u = torch.ones(2, 3)
    s = torch.full((2, 3), 0.5)
    m = torch.full((2, 3), -0.5)

    for bad in (
        lambda: solve([[0.0, 1.0]], u, s, m, 1, 10),
        lambda: solve(c, torch.ones(3, 3), s, m, 1, 10),
        lambda: solve(c, u, torch.ones(2, 2), m, 1, 10),
        lambda: solve(c, u, s, torch.ones(1, 6), 1, 10),
        lambda: solve(torch.tensor(1.0), None, None, None, 1, 10),
        lambda: solve(c, u, s, m, -1, 10),
        lambda: solve(c, u, s, m, 10, 10),
        lambda: solve(c, u, s, m, 1.5, 10),
        lambda: solve(c, u, s, m, True, 10),
        lambda: solve(c, u, s, m, 1, 0),
        lambda: solve(c, u, s, m, 1, 10, cfg_scale=float("nan")),
        lambda: solve(c, u, s, m, 1, 10, stg_scale=float("inf")),
        lambda: solve(c, u, s, m, 1, 10, modality_scale="1.5"),
        lambda: solve(c, u, s, m, 1, 10, start=-0.1),
        lambda: solve(c, u, s, m, 1, 10, stop=1.2),
        lambda: solve(c, u, s, m, 1, 10, start=0.9, stop=0.1),
        # 分量生效却缺少对应预测
        lambda: solve(c, None, s, m, 1, 10, cfg_scale=3.0),
        lambda: solve(c, u, None, m, 1, 10, stg_scale=1.0),
        lambda: solve(c, u, s, None, 1, 10, modality_scale=2.0),
    ):
        try:
            bad()
            raise AssertionError("非法输入应抛 ValueError")
        except ValueError:
            pass
