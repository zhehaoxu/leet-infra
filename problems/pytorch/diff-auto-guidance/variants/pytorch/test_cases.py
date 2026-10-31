import math

import torch


C = torch.tensor([[1.0, 2.0, 3.0, 4.0], [0.0, 1.0, -1.0, 2.0]], dtype=torch.float64)
U = torch.tensor([[0.2, 1.0, 2.5, 3.0], [0.5, 0.0, 0.5, -0.5]], dtype=torch.float64)


def _std(row):
    n = len(row)
    mean = sum(row) / n
    return math.sqrt(sum((x - mean) ** 2 for x in row) / (n - 1))


def _oracle(cond, uncond, step, steps, scale, start, stop, original, rescale):
    """独立 oracle：逐元素纯 Python 重做整套语义（含无偏标准差）。"""
    within = int(start * steps) <= step < int(stop * steps)
    close = math.isclose(scale, 0.0) if original else math.isclose(scale, 1.0)
    rows = []
    for rc, ru in zip(cond.tolist(), uncond.tolist()):
        if within and not close:
            out = [(c if original else u) + scale * (c - u) for c, u in zip(rc, ru)]
        else:
            out = list(rc)
        if rescale > 0.0:
            ratio = _std(rc) / _std(out)
            out = [rescale * (x * ratio) + (1.0 - rescale) * x for x in out]
        rows.append(out)
    return rows


def _close(got, want_rows):
    return torch.allclose(
        got.to(torch.float64), torch.tensor(want_rows, dtype=torch.float64), rtol=1e-5, atol=1e-6
    )


def check_example(solve):
    got = solve(C, U, 5, 20, 7.5, 0.1, 0.8)
    assert [[round(float(v), 6) for v in row] for row in got.tolist()] == [
        [6.2, 8.5, 6.25, 10.5],
        [-3.25, 7.5, -10.75, 18.25],
    ], got.tolist()
    assert got.dtype == torch.float32 and tuple(got.shape) == (2, 4)

    flip = solve(C, U, 5, 20, 7.5, 0.1, 0.8, True)
    assert [[round(float(v), 6) for v in row] for row in flip.tolist()] == [
        [7.0, 9.5, 6.75, 11.5],
        [-3.75, 8.5, -12.25, 20.75],
    ], flip.tolist()

    half = solve(C, U, 5, 20, 7.5, 0.1, 0.8, False, 0.5)
    assert [[round(float(v), 6) for v in row] for row in half.tolist()] == [
        [5.043058, 6.91387, 5.083728, 8.540663],
        [-1.790692, 4.132366, -5.923058, 10.055424],
    ], half.tolist()

    outside = solve(C, U, 17, 20, 7.5, 0.1, 0.8)
    assert torch.equal(outside, C.to(torch.float32)), outside.tolist()

    zero_scale = solve(C, U, 5, 20, 0.0)
    assert torch.equal(zero_scale, U.to(torch.float32)), zero_scale.tolist()


check_example.example = True


def check_identity_thresholds_flip(solve):
    # 默认表述：1.0 是恒等，0.0 不是（0.0 把锚点推到 pred_uncond 上）
    assert torch.equal(solve(C, U, 5, 20, 1.0), C.to(torch.float32))
    assert torch.equal(solve(C, U, 5, 20, 0.0), U.to(torch.float32))
    # 原论文表述：0.0 是恒等，1.0 不是（1.0 等于再外推一个 shift）
    assert torch.equal(solve(C, U, 5, 20, 0.0, 0.0, 1.0, True), C.to(torch.float32))
    assert _close(solve(C, U, 5, 20, 1.0, 0.0, 1.0, True), [[1.8, 3.0, 3.5, 5.0], [-0.5, 2.0, -2.5, 4.5]])


def check_isclose_is_relative(solve):
    # 与 1.0 比是相对容差：1+1e-10 仍算恒等 -> 关
    assert torch.equal(solve(C, U, 5, 20, 1.0 + 1e-10), C.to(torch.float32))
    # 1+1e-8 超出 rel_tol=1e-9 -> 开，但输出仍几乎等于 pred_cond
    got = solve(C, U, 5, 20, 1.0 + 1e-8)
    assert _close(got, C.tolist())
    # 与 0.0 比是绝对容差（abs_tol 默认 0）：原论文表述下 1e-12 不算恒等 -> 开
    off_ref = C.to(torch.float32)
    got2 = solve(C, U, 5, 20, 1e-12, 0.0, 1.0, True)
    assert not torch.equal(got2, off_ref)
    assert torch.allclose(got2, off_ref, rtol=0, atol=1e-11), got2.tolist()
    # 恰好 0.0 才算恒等
    assert torch.equal(solve(C, U, 5, 20, 0.0, 0.0, 1.0, True), off_ref)


def check_window_is_left_closed_right_open(solve):
    steps = 20
    first_on, last_on = int(0.1 * steps), int(0.8 * steps) - 1
    assert first_on == 2 and last_on == 15
    cond = C.to(torch.float32)
    assert not torch.equal(solve(C, U, first_on, steps, 7.5, 0.1, 0.8), cond)
    assert not torch.equal(solve(C, U, last_on, steps, 7.5, 0.1, 0.8), cond)
    assert torch.equal(solve(C, U, last_on + 1, steps, 7.5, 0.1, 0.8), cond)
    assert torch.equal(solve(C, U, first_on - 1, steps, 7.5, 0.1, 0.8), cond)
    # int 截断是向下取整，不是四舍五入：int(0.25*14)=3
    assert torch.equal(solve(C, U, 2, 14, 7.5, 0.25, 0.5), cond)
    assert not torch.equal(solve(C, U, 3, 14, 7.5, 0.25, 0.5), cond)
    # 窗口可以在截断后塌成空集：int(0.93*10)=9 <= step < int(0.97*10)=9
    assert torch.equal(solve(C, U, 9, 10, 7.5, 0.93, 0.97), cond)


def check_uncond_optional(solve):
    cond = C.to(torch.float32)
    # 关闭的三种来源：窗口外 / 恒等尺度 / 二者叠加，都可以不传 pred_uncond
    assert torch.equal(solve(C, None, 19, 20, 7.5, 0.1, 0.8), cond)
    assert torch.equal(solve(C, None, 5, 20, 1.0), cond)
    assert torch.equal(solve(C, None, 19, 20, 1.0), cond)
    # 关闭分支根本不读 pred_uncond：给它一个形状都不符的垃圾张量也不应报错
    assert torch.equal(solve(C, torch.zeros(1), 19, 20, 7.5, 0.1, 0.8), cond)
    assert torch.equal(solve(C, torch.zeros(1), 5, 20, 1.0), cond)
    # 生效却不传 -> ValueError
    try:
        solve(C, None, 5, 20, 7.5)
    except ValueError:
        pass
    else:
        raise AssertionError("生效分支缺 pred_uncond 应抛 ValueError")


def check_rescale_runs_on_off_branch(solve):
    cond = C.to(torch.float32)
    # r=0.5：比例是同一个标准差之比，恰好为 1，且 0.5x+0.5x 无舍入损失
    assert torch.equal(solve(C, U, 19, 20, 7.5, 0.1, 0.8, False, 0.5), cond)
    assert torch.equal(solve(C, U, 5, 20, 1.0, 0.0, 1.0, False, 0.3), cond)
    # 非 0.5 的混合系数可能带来 1 ulp 级别的抖动，只能按容差比
    got = solve(C, U, 19, 20, 7.5, 0.1, 0.8, False, 0.3)
    assert torch.allclose(got, cond, rtol=0, atol=1e-7), got.tolist()
    # r=0 才是真正跳过
    assert torch.equal(solve(C, U, 5, 20, 7.5, 0.0, 1.0, False, 0.0), solve(C, U, 5, 20, 7.5))


def check_rescale_restores_std(solve):
    raw = solve(C, U, 5, 20, 7.5, 0.0, 1.0, False, 0.0)
    got = solve(C, U, 5, 20, 7.5, 0.0, 1.0, False, 1.0)
    ratio = got.to(torch.float64).std(dim=1) / C.std(dim=1)
    assert torch.allclose(ratio, torch.ones(2, dtype=torch.float64), atol=1e-6), ratio.tolist()
    # 0 < r < 1 时按比例部分回收，逐样本独立
    shrink = raw.to(torch.float64).std(dim=1, keepdim=True) / C.std(dim=1, keepdim=True)
    assert (shrink > 1.0).all(), shrink.tolist()
    part = solve(C, U, 5, 20, 7.5, 0.0, 1.0, False, 0.5)
    want = 0.5 * (raw.to(torch.float64) / shrink) + 0.5 * raw.to(torch.float64)
    assert torch.allclose(part.to(torch.float64), want, rtol=1e-6, atol=1e-6), part.tolist()


def check_per_sample_reduction(solve):
    c = torch.stack([C, C * 10.0])  # (2, 2, 4)
    u = torch.stack([U, U * 10.0])
    got = solve(c, u, 5, 20, 3.0, 0.0, 1.0, False, 0.7)
    plain = solve(c, u, 5, 20, 3.0, 0.0, 1.0)
    dims = [1, 2]
    std_c = c.to(torch.float64).std(dim=dims, keepdim=True)
    std_p = plain.to(torch.float64).std(dim=dims, keepdim=True)
    want = 0.7 * (plain.to(torch.float64) * (std_c / std_p)) + 0.3 * plain.to(torch.float64)
    assert torch.allclose(got.to(torch.float64), want, rtol=1e-5, atol=1e-4), got.tolist()
    # 第 0 维（batch）不参与归约：单独看第一样本
    one = solve(c[:1], u[:1], 5, 20, 3.0, 0.0, 1.0, False, 0.7)
    assert torch.allclose(one.to(torch.float64), want[:1], rtol=1e-5, atol=1e-4)


def check_against_oracle(solve):
    torch.manual_seed(7)
    for _ in range(120):
        steps = int(torch.randint(1, 31, (1,)))
        step = int(torch.randint(0, steps, (1,)))
        start = float(torch.randint(0, 9, (1,))) / 10.0
        stop = min(1.0, start + float(torch.randint(1, 11, (1,))) / 10.0)
        scale = float(torch.randint(-30, 80, (1,))) / 10.0
        if scale == 1.0 or scale == 0.0:
            scale = 2.5
        original = bool(int(torch.randint(0, 2, (1,))))
        rescale = float(torch.randint(0, 11, (1,))) / 10.0
        c = torch.randn(3, 5, dtype=torch.float64)
        u = torch.randn(3, 5, dtype=torch.float64)
        got = solve(c, u, step, steps, scale, start, stop, original, rescale)
        assert _close(got, _oracle(c, u, step, steps, scale, start, stop, original, rescale)), (
            step,
            steps,
            start,
            stop,
            scale,
            original,
            rescale,
        )


def check_dtypes_and_float32_input(solve):
    c32 = C.to(torch.float32)
    u32 = U.to(torch.float32)
    for c, u in ((c32, u32), (C, U), (C, u32), (c32, U)):
        got = solve(c, u, 5, 20, 7.5, 0.1, 0.8, False, 0.4)
        assert got.dtype == torch.float32 and tuple(got.shape) == (2, 4)
        assert torch.isfinite(got).all()
        assert _close(got, _oracle(C, U, 5, 20, 7.5, 0.1, 0.8, False, 0.4))
    # 3 维输入同样返回原形状
    got = solve(C.unsqueeze(-1), U.unsqueeze(-1), 5, 20, 7.5, 0.1, 0.8, True, 0.6)
    assert tuple(got.shape) == (2, 4, 1) and got.dtype == torch.float32


def _poison(t, value):
    out = t.clone()
    out[0, 0] = value
    return out


def check_invalid_inputs(solve):
    bad = [
        ("pred_cond 为 None", lambda: solve(None, U, 0, 4)),
        ("pred_cond 非张量", lambda: solve([[1.0, 2.0]], U, 0, 4)),
        ("pred_cond 只有一维", lambda: solve(torch.tensor([1.0, 2.0, 3.0]), None, 0, 4)),
        ("单样本只有 1 个元素", lambda: solve(torch.tensor([[1.0], [2.0]]), None, 0, 4)),
        ("pred_uncond 非张量", lambda: solve(C, [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], 5, 20, 7.5)),
        ("pred_uncond 形状不同", lambda: solve(C, U[:, :3], 5, 20, 7.5)),
        ("pred_cond 含 NaN", lambda: solve(_poison(C, float("nan")), U, 0, 4)),
        ("pred_cond 含 inf", lambda: solve(_poison(C, float("inf")), U, 0, 4)),
        ("pred_uncond 含 NaN", lambda: solve(C, _poison(U, float("nan")), 5, 20, 7.5)),
        ("step 为浮点", lambda: solve(C, U, 1.0, 4)),
        ("step 为 bool", lambda: solve(C, U, True, 4)),
        ("step 为负", lambda: solve(C, U, -1, 4)),
        ("step 越界", lambda: solve(C, U, 4, 4)),
        ("N 为浮点", lambda: solve(C, U, 0, 4.0)),
        ("N 为 bool", lambda: solve(C, U, 0, True)),
        ("N 为 0", lambda: solve(C, U, 0, 0)),
        ("N 为负", lambda: solve(C, U, 0, -3)),
        ("scale 为 NaN", lambda: solve(C, U, 5, 20, float("nan"))),
        ("scale 为 inf", lambda: solve(C, U, 5, 20, float("inf"))),
        ("scale 非实数", lambda: solve(C, U, 5, 20, "7.5")),
        ("start 为 1.0", lambda: solve(C, U, 0, 4, 7.5, 1.0, 1.0)),
        ("start 为负", lambda: solve(C, U, 0, 4, 7.5, -0.1, 1.0)),
        ("start 大于 stop", lambda: solve(C, U, 0, 4, 7.5, 0.7, 0.3)),
        ("stop 大于 1", lambda: solve(C, U, 0, 4, 7.5, 0.0, 1.5)),
        ("stop 为 NaN", lambda: solve(C, U, 0, 4, 7.5, 0.0, float("nan"))),
        ("start 非实数", lambda: solve(C, U, 0, 4, 7.5, None, 1.0)),
        ("use_original_formulation 为 int", lambda: solve(C, U, 5, 20, 7.5, 0.0, 1.0, 1)),
        ("use_original_formulation 为 None", lambda: solve(C, U, 5, 20, 7.5, 0.0, 1.0, None)),
        ("rescale 为负", lambda: solve(C, U, 5, 20, 7.5, 0.0, 1.0, False, -0.1)),
        ("rescale 大于 1", lambda: solve(C, U, 5, 20, 7.5, 0.0, 1.0, False, 1.5)),
        ("rescale 为 NaN", lambda: solve(C, U, 5, 20, 7.5, 0.0, 1.0, False, float("nan"))),
    ]
    for name, fn in bad:
        try:
            fn()
        except ValueError:
            continue
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(f"{name}: 抛了 {type(exc).__name__}，应为 ValueError") from exc
        raise AssertionError(f"{name}: 未抛 ValueError")
