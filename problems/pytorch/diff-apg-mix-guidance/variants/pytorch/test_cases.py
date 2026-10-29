import math

import torch


def _rows(tensor):
    return tensor.reshape(tensor.shape[0], -1).tolist()


def _oracle(cond, uncond, buffer, step, steps, cfg, apg, mu, thr, eta, apg_start, start, stop, orig):
    """独立 oracle：窗口、动量、截断、投影全部用 python 标量重做。"""
    off = 0.0 if orig else 1.0
    cfg_on = int(start * steps) <= step < int(stop * steps) and not math.isclose(cfg, off)
    apg_on = cfg_on and step > apg_start and not math.isclose(apg, off)

    rows_c = _rows(cond)
    rows_u = _rows(uncond)
    rows_b = _rows(buffer) if buffer is not None else None

    out, new_buffer = [], []
    for i, (rc, ru) in enumerate(zip(rows_c, rows_u)):
        if not cfg_on:
            out.append(rc)
            new_buffer.append(None if mu is None or rows_b is None else rows_b[i])
            continue

        diff = [ci - ui for ci, ui in zip(rc, ru)]
        if mu is None:
            running = list(diff)
        else:
            prev = rows_b[i] if rows_b is not None else [0.0] * len(rc)
            running = [di + mu * pi for di, pi in zip(diff, prev)]
        new_buffer.append(running)

        base = rc if orig else ru
        if not apg_on:
            out.append([b + cfg * di for b, di in zip(base, diff)])
            continue

        if thr > 0.0:
            norm = math.sqrt(sum(x * x for x in running))
            factor = min(1.0, thr / max(norm, 1e-12))
            running = [x * factor for x in running]

        cond_norm = max(math.sqrt(sum(x * x for x in rc)), 1e-12)
        direction = [x / cond_norm for x in rc]
        parallel_scalar = sum(x * y for x, y in zip(running, direction))
        upd = []
        for d, u in zip(running, direction):
            par = parallel_scalar * u
            upd.append((d - par) + eta * par)
        out.append([b + apg * x for b, x in zip(base, upd)])

    shape = cond.shape
    pred = torch.tensor(out, dtype=torch.float64).reshape(shape).to(torch.float32)
    if mu is None or any(row is None for row in new_buffer):
        return pred, None
    buf = torch.tensor(new_buffer, dtype=torch.float64).reshape(shape).to(torch.float32)
    return pred, buf


COND = torch.tensor([[0.4, -1.2, 0.6], [0.0, 0.5, -0.5]])
UNCOND = torch.tensor([[0.1, -0.4, 0.2], [0.2, -0.1, 0.0]])


def check_example(solve):
    # step=3：APG 尚未开始（3 > 5 不成立），走 CFG，动量被写成 d
    pred3, buf3 = solve(COND, UNCOND, 3, 10)
    assert [[round(float(v), 6) for v in row] for row in pred3.tolist()] == [
        [1.15, -3.2, 1.6],
        [-0.5, 2.0, -1.75],
    ], pred3.tolist()
    assert [[round(float(v), 6) for v in row] for row in buf3.tolist()] == [
        [0.3, -0.8, 0.4],
        [-0.2, 0.6, -0.5],
    ], buf3.tolist()

    # step=6：APG 开启，动量 = d6 + (-0.5)*d3，再投影到 cond 的正交补
    pred6, buf6 = solve(COND, UNCOND, 6, 10, buffer=buf3)
    assert [[round(float(v), 6) for v in row] for row in pred6.tolist()] == [
        [0.253061, -0.359184, 0.179592],
        [-0.8, 0.15, 0.25],
    ], pred6.tolist()
    assert [[round(float(v), 6) for v in row] for row in buf6.tolist()] == [
        [0.15, -0.4, 0.2],
        [-0.1, 0.3, -0.25],
    ], buf6.tolist()

    # 同样 step=6 但从零起：动量就是 d6，投影结果不同
    cold, cold_buf = solve(COND, UNCOND, 6, 10)
    assert [[round(float(v), 6) for v in row] for row in cold.tolist()] == [
        [0.406122, -0.318367, 0.159184],
        [-1.8, 0.4, 0.5],
    ], cold.tolist()
    assert torch.allclose(cold_buf, buf3, atol=1e-6), "buffer=None 时动量退化为当前差值"

    assert pred6.dtype == torch.float32 and pred6.shape == COND.shape and buf6.dtype == torch.float32


check_example.example = True


def check_warmup_phase(solve):
    torch.manual_seed(0)
    c = torch.randn(2, 4, dtype=torch.float64)
    u = torch.randn(2, 4, dtype=torch.float64)

    # apg_start_step=5：step=0..5 都只做 CFG，但动量每步都在累积
    buf = None
    for step in range(6):
        pred, buf = solve(c, u, step, 10, buffer=buf)
        assert torch.allclose(pred, (u + 3.5 * (c - u)).to(torch.float32), atol=1e-6), f"step={step} 应等于 CFG"

    prev = None
    buf = None
    for step in range(6):
        _, buf = solve(c, u, step, 10, buffer=buf)
        d = c - u
        want = d if prev is None else d - 0.5 * prev
        assert torch.allclose(buf.to(torch.float64), want, atol=1e-5), f"step={step} 动量递推不对"
        prev = buf.to(torch.float64)

    # step=6 起换成 APG，结果不再等于 CFG
    apg_pred, _ = solve(c, u, 6, 10, buffer=buf)
    assert not torch.allclose(apg_pred, (u + 3.5 * (c - u)).to(torch.float32), atol=1e-3)
    # APG 关闭的那一步若 momentum=None，则 buffer 恒为 None
    _, none_buf = solve(c, u, 0, 10, momentum=None)
    assert none_buf is None


def check_momentum_variants(solve):
    torch.manual_seed(1)
    c = torch.randn(2, 5, dtype=torch.float64)
    u = torch.randn(2, 5, dtype=torch.float64)

    # momentum=None：无动量，buffer 恒为 None，APG 直接用当前差值
    pred, buf = solve(c, u, 7, 10, momentum=None)
    assert buf is None
    d = c - u
    norm = d.norm(dim=(1,), keepdim=True).clamp_min(1e-12)
    d_cut = d * torch.clamp(10.0 / norm, max=1.0)
    direction = c / c.norm(dim=(1,), keepdim=True).clamp_min(1e-12)
    parallel = (d_cut * direction).sum(dim=1, keepdim=True) * direction
    want = u + 10.0 * (d_cut - parallel)
    assert torch.allclose(pred, want.to(torch.float32), atol=1e-5)

    # momentum=0.0：running 就等于当前差值，与 momentum=None 结果相同
    pred0, buf0 = solve(c, u, 7, 10, momentum=0.0)
    assert torch.allclose(pred0, pred, atol=1e-6)
    assert torch.allclose(buf0.to(torch.float64), d, atol=1e-6)

    # 负动量：running = d - 0.5*prev，方向被上一拍对冲
    prev = torch.full_like(c, 0.7)
    pred_m, buf_m = solve(c, u, 7, 10, buffer=prev, momentum=-0.5)
    assert torch.allclose(buf_m.to(torch.float64), d - 0.5 * prev, atol=1e-5)

    # 输入 buffer 不被就地改写
    sentinel = prev.clone()
    solve(c, u, 7, 10, buffer=sentinel)
    assert torch.equal(sentinel, prev)


def check_norm_threshold(solve):
    torch.manual_seed(2)
    c = torch.randn(2, 6, dtype=torch.float64)
    u = torch.randn(2, 6, dtype=torch.float64)

    # threshold=0.0 关闭截断：等价于用无穷大阈值
    a, _ = solve(c, u, 7, 10, norm_threshold=0.0, momentum=None)
    b, _ = solve(c, u, 7, 10, norm_threshold=1e9, momentum=None)
    assert torch.allclose(a, b, atol=1e-6)

    # 阈值变小，更新方向的模长被压到阈值以内
    big, _ = solve(c, u, 7, 10, norm_threshold=100.0, momentum=None)
    small, _ = solve(c, u, 7, 10, norm_threshold=0.05, momentum=None)
    d = (c - u)
    assert d.norm(dim=1).min() > 0.05, "用例应触发截断"
    base = u
    upd_big = (big.to(torch.float64) - base) / 10.0
    upd_small = (small.to(torch.float64) - base) / 10.0
    assert upd_small.norm(dim=1).max() <= 0.05 + 1e-6, "截断后模长不应超过阈值"
    assert upd_big.norm(dim=1).min() > 0.05, "大阈值不应触发截断"
    # 截断作用在差值上，因此小阈值的结果方向与未截断一致
    cos = (upd_small * upd_big).sum(dim=1) / (upd_small.norm(dim=1) * upd_big.norm(dim=1))
    assert torch.allclose(cos, torch.ones_like(cos), atol=1e-5), "截断只缩放不改方向"


def check_projection(solve):
    torch.manual_seed(3)
    c = torch.randn(3, 4, 5, dtype=torch.float64)
    u = torch.randn(3, 4, 5, dtype=torch.float64)

    # eta=1：normalized 等于 (截断后的) 差值本身，即退化成普通 CFG（基准仍是 uncond）
    full, _ = solve(c, u, 8, 10, eta=1.0, momentum=None, norm_threshold=0.0)
    assert torch.allclose(full, (u + 10.0 * (c - u)).to(torch.float32), atol=1e-5)

    # eta=0：更新方向与 cond 逐样本正交
    orth, _ = solve(c, u, 8, 10, eta=0.0, momentum=None, norm_threshold=0.0)
    upd = (orth.to(torch.float64) - u) / 10.0
    dots = (upd * c).sum(dim=(1, 2))
    assert torch.allclose(dots, torch.zeros_like(dots), atol=1e-5), "eta=0 时更新应与 cond 正交"
    # 正交分解只会缩短向量
    assert (upd.norm(dim=(1, 2)) <= (c - u).norm(dim=(1, 2)) + 1e-6).all()

    # eta=0.5 落在两者之间
    half, _ = solve(c, u, 8, 10, eta=0.5, momentum=None, norm_threshold=0.0)
    upd_half = (half.to(torch.float64) - u) / 10.0
    assert torch.allclose(upd_half, (upd + 0.5 * ((c - u) - upd)), atol=1e-5)


def check_window_and_formulation(solve):
    torch.manual_seed(4)
    c = torch.randn(2, 4, dtype=torch.float64)
    u = torch.randn(2, 4, dtype=torch.float64)

    # start=0.5 / stop=1.0：step=4 在窗口外，直接回传 cond 且不更新动量
    prev = torch.full_like(c, 0.3)
    pred, buf = solve(c, u, 4, 10, buffer=prev, start=0.5, stop=1.0)
    assert torch.allclose(pred, c.to(torch.float32), atol=1e-7)
    assert torch.allclose(buf.to(torch.float64), prev, atol=1e-6), "窗口外不应更新动量"

    # cfg_scale=1.0：CFG 关闭，APG 也随之关闭（apg_on 要求 cfg_on）
    plain, _ = solve(c, u, 8, 10, cfg_scale=1.0, momentum=None)
    assert torch.allclose(plain, c.to(torch.float32), atol=1e-7)

    # apg_scale=1.0：只有 APG 关闭，仍走 CFG
    cfg_only, _ = solve(c, u, 8, 10, apg_scale=1.0, momentum=None)
    assert torch.allclose(cfg_only, (u + 3.5 * (c - u)).to(torch.float32), atol=1e-5)

    # use_original_formulation=True：基准换成 cond，且关闭判据换成 isclose(s, 0.0)
    orig_off, _ = solve(c, u, 8, 10, cfg_scale=0.0, momentum=None, use_original_formulation=True)
    assert torch.allclose(orig_off, c.to(torch.float32), atol=1e-7)
    orig_cfg, _ = solve(c, u, 3, 10, momentum=None, use_original_formulation=True)
    assert torch.allclose(orig_cfg, (c + 3.5 * (c - u)).to(torch.float32), atol=1e-5)
    orig_apg, _ = solve(c, u, 8, 10, momentum=None, use_original_formulation=True, norm_threshold=0.0, eta=1.0)
    assert torch.allclose(orig_apg, (c + 10.0 * (c - u)).to(torch.float32), atol=1e-5)


def check_against_oracle(solve):
    torch.manual_seed(5)
    c = torch.randn(2, 3, 4, dtype=torch.float64)
    u = torch.randn(2, 3, 4, dtype=torch.float64)

    settings = [
        dict(cfg_scale=3.5, apg_scale=10.0, momentum=-0.5, norm_threshold=10.0, eta=0.0, apg_start_step=5),
        dict(cfg_scale=4.0, apg_scale=2.0, momentum=0.0, norm_threshold=0.1, eta=0.7, apg_start_step=0),
        dict(cfg_scale=6.0, apg_scale=8.0, momentum=0.5, norm_threshold=0.0, eta=1.0, apg_start_step=2),
        dict(
            cfg_scale=2.5,
            apg_scale=12.0,
            momentum=None,
            norm_threshold=0.3,
            eta=0.25,
            apg_start_step=4,
            use_original_formulation=True,
        ),
    ]

    for cfg in settings:
        buf = None
        for step in range(10):
            got, got_buf = solve(c, u, step, 10, buffer=buf, **cfg)
            want, want_buf = _oracle(
                c,
                u,
                buf,
                step,
                10,
                cfg["cfg_scale"],
                cfg["apg_scale"],
                cfg["momentum"],
                cfg["norm_threshold"],
                cfg["eta"],
                cfg["apg_start_step"],
                0.0,
                1.0,
                cfg.get("use_original_formulation", False),
            )
            assert torch.allclose(got, want, atol=1e-5), f"{cfg} step={step} 预测不一致"
            if cfg["momentum"] is None:
                assert got_buf is None and want_buf is None
            elif want_buf is None:
                assert got_buf is None, "窗口外且无初始 buffer 时应回传 None"
            else:
                assert torch.allclose(got_buf, want_buf, atol=1e-5), f"{cfg} step={step} 动量不一致"
            buf = got_buf

    # 窗口不是全区间时也对拍一遍
    buf = None
    for step in range(10):
        got, got_buf = solve(c, u, step, 10, buffer=buf, start=0.3, stop=0.9)
        want, want_buf = _oracle(c, u, buf, step, 10, 3.5, 10.0, -0.5, 10.0, 0.0, 5, 0.3, 0.9, False)
        assert torch.allclose(got, want, atol=1e-5)
        if want_buf is None:
            assert got_buf is None
        else:
            assert torch.allclose(got_buf, want_buf, atol=1e-5)
            buf = got_buf


def check_invalid(solve):
    c = torch.zeros(2, 3)
    u = torch.ones(2, 3)

    for bad in (
        lambda: solve([[0.0, 1.0]], u, 3, 10),
        lambda: solve(c, torch.ones(3, 3), 3, 10),
        lambda: solve(c, u, 3, 10, buffer=torch.zeros(2, 2)),
        lambda: solve(c, u, 3, 10, buffer=torch.zeros(4, 3)),
        lambda: solve(c, u, -1, 10),
        lambda: solve(c, u, 10, 10),
        lambda: solve(c, u, 1.5, 10),
        lambda: solve(c, u, True, 10),
        lambda: solve(c, u, 3, 0),
        lambda: solve(c, u, 3, 10, cfg_scale=-1.0),
        lambda: solve(c, u, 3, 10, apg_scale=-1.0),
        lambda: solve(c, u, 3, 10, norm_threshold=-0.5),
        lambda: solve(c, u, 3, 10, eta=-0.5),
        lambda: solve(c, u, 3, 10, momentum=float("nan")),
        lambda: solve(c, u, 3, 10, apg_start_step=-1),
        lambda: solve(c, u, 3, 10, apg_start_step=2.5),
        lambda: solve(c, u, 3, 10, start=-0.1),
        lambda: solve(c, u, 3, 10, stop=1.5),
        lambda: solve(c, u, 3, 10, start=0.8, stop=0.2),
        lambda: solve(c, u, 3, 10, buffer=torch.zeros(2, 3), momentum=None),
        lambda: solve(c, None, 3, 10),
        lambda: solve(c, u, 3, 10, cfg_scale="3.5"),
    ):
        try:
            bad()
            raise AssertionError("非法输入应抛 ValueError")
        except ValueError:
            pass
