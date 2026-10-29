import math

import numpy as np
import torch

ABAR = np.array([0.99, 0.9, 0.7, 0.5, 0.3, 0.15, 0.06, 0.02, 0.005, 0.001], dtype=np.float64)


def _grid_oracle(train, steps, spacing, offset):
    if spacing == "linspace":
        return np.linspace(0, train - 1, steps, dtype=np.float32)[::-1].copy()
    if spacing == "leading":
        ratio = train // steps
        return (np.arange(0, steps) * ratio).round()[::-1].copy().astype(np.float32) + offset
    ratio = train / steps
    return np.arange(train, 0, -ratio).round().copy().astype(np.float32) - 1


def _sigma_to_t_oracle(sigma, log_sigmas):
    log_sigma = math.log(max(sigma, 1e-10))
    dists = log_sigma - log_sigmas
    low = int(np.cumsum(dists >= 0).argmax())
    low = min(low, len(log_sigmas) - 2)
    high = low + 1
    w = (log_sigmas[low] - log_sigma) / (log_sigmas[low] - log_sigmas[high])
    w = min(max(w, 0.0), 1.0)
    return (1.0 - w) * low + w * high


def _oracle(
    alphas,
    steps,
    spacing="linspace",
    interpolation="linear",
    karras=False,
    exponential=False,
    final="zero",
    sigma_min=None,
    sigma_max=None,
    offset=0,
):
    """独立 oracle：numpy 的 linspace/arange/round/interp + 标量反查，逐元素重做整条流水线。"""
    alphas = np.asarray(alphas, dtype=np.float64)
    train = alphas.size
    grid = _grid_oracle(train, steps, spacing, offset)
    sigmas_full = np.sqrt((1.0 - alphas) / alphas)
    log_sigmas = np.log(sigmas_full)

    if interpolation == "linear":
        sigmas = np.interp(grid, np.arange(0, train), sigmas_full)
    else:
        sigmas = np.exp(np.linspace(math.log(sigmas_full[-1]), math.log(sigmas_full[0]), steps + 1))

    if karras or exponential:
        # 参考解里的 in_sigmas 是从噪声端往干净端递减的，所以 [0] 是 σ_max、[-1] 是 σ_min
        lo = sigmas_full[0] if sigma_min is None else sigma_min
        hi = sigmas_full[-1] if sigma_max is None else sigma_max
        if karras:
            ramp = np.linspace(0.0, 1.0, steps)
            sigmas = (hi ** (1 / 7.0) + ramp * (lo ** (1 / 7.0) - hi ** (1 / 7.0))) ** 7.0
        else:
            sigmas = np.exp(np.linspace(math.log(hi), math.log(lo), steps))
        grid = np.array([_sigma_to_t_oracle(s, log_sigmas) for s in sigmas], dtype=np.float32)

    last = 0.0 if final == "zero" else math.sqrt((1.0 - alphas[0]) / alphas[0])
    sigmas = np.concatenate([np.asarray(sigmas, dtype=np.float64), [last]])
    return grid.astype(np.float32), sigmas.astype(np.float32)


def _check(
    solve,
    abar=ABAR,
    steps=3,
    spacing="linspace",
    interpolation="linear",
    karras=False,
    exponential=False,
    final="zero",
    sigma_min=None,
    sigma_max=None,
    offset=0,
):
    got_t, got_s = solve(
        torch.tensor(abar, dtype=torch.float32),
        steps,
        timestep_spacing=spacing,
        interpolation_type=interpolation,
        use_karras_sigmas=karras,
        use_exponential_sigmas=exponential,
        final_sigmas_type=final,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        steps_offset=offset,
    )
    want_t, want_s = _oracle(
        abar,
        steps,
        spacing=spacing,
        interpolation=interpolation,
        karras=karras,
        exponential=exponential,
        final=final,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        offset=offset,
    )
    assert got_t.dtype == torch.float32 and got_s.dtype == torch.float32
    assert got_t.shape == want_t.shape, (got_t.shape, want_t.shape)
    assert got_s.shape == want_s.shape, (got_s.shape, want_s.shape)
    assert np.allclose(got_t.numpy(), want_t, atol=1e-5), (got_t.tolist(), want_t.tolist())
    assert np.allclose(got_s.numpy(), want_s, rtol=1e-6, atol=1e-6), (got_s.tolist(), want_s.tolist())
    return got_t, got_s


def check_example(solve):
    t, s = solve(torch.tensor(ABAR, dtype=torch.float32), 3)
    assert [round(float(v), 6) for v in t.tolist()] == [9.0, 4.5, 0.0], t.tolist()
    assert [round(float(v), 6) for v in s.tolist()] == [31.606960, 1.954001, 0.100504, 0.0], s.tolist()

    t, s = solve(torch.tensor(ABAR, dtype=torch.float32), 3, timestep_spacing="leading")
    assert [round(float(v), 6) for v in t.tolist()] == [6.0, 3.0, 0.0]
    assert [round(float(v), 6) for v in s.tolist()] == [3.958114, 1.0, 0.100504, 0.0]

    t, s = solve(torch.tensor(ABAR, dtype=torch.float32), 3, timestep_spacing="trailing")
    assert [round(float(v), 6) for v in t.tolist()] == [9.0, 6.0, 2.0]
    assert [round(float(v), 6) for v in s.tolist()] == [31.606960, 3.958114, 0.654654, 0.0]

    t, s = solve(torch.tensor(ABAR, dtype=torch.float32), 3, use_karras_sigmas=True)
    assert [round(float(v), 6) for v in t.tolist()] == [9.0, 5.561106, 0.0], t.tolist()
    assert [round(float(v), 6) for v in s.tolist()] == [31.606960, 3.166429, 0.100504, 0.0], s.tolist()

    t, s = solve(torch.tensor(ABAR, dtype=torch.float32), 3, use_exponential_sigmas=True)
    assert [round(float(v), 6) for v in t.tolist()] == [9.0, 4.347704, 0.0], t.tolist()
    assert [round(float(v), 6) for v in s.tolist()] == [31.606960, 1.782307, 0.100504, 0.0]


check_example.example = True


def check_spacing_variants(solve):
    for steps in (1, 3, 4, 7):
        for spacing in ("linspace", "leading", "trailing"):
            _check(solve, steps=steps, spacing=spacing)

    # T=9 / N=4：leading 用整除 9//4=2，trailing 用真除 9/4=2.25
    small = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    t_lead, _ = _check(solve, abar=small, steps=4, spacing="leading")
    assert t_lead.tolist() == [6.0, 4.0, 2.0, 0.0], t_lead.tolist()
    t_trail, _ = _check(solve, abar=small, steps=4, spacing="trailing")
    assert t_trail.tolist() == [8.0, 6.0, 3.0, 1.0], t_trail.tolist()
    # 4.5 落在 banker's rounding 的偶数侧，round(4.5) = 4 而不是 5
    assert np.round(4.5) == 4.0 and np.round(2.5) == 2.0


def check_leading_offset(solve):
    base, _ = _check(solve, steps=4, spacing="leading")
    shifted, _ = _check(solve, steps=4, spacing="leading", offset=10)
    assert torch.allclose(shifted, base + 10.0)
    # steps_offset 只作用于 leading
    for spacing in ("linspace", "trailing"):
        plain, _ = _check(solve, steps=4, spacing=spacing)
        with_offset, _ = _check(solve, steps=4, spacing=spacing, offset=10)
        assert torch.equal(plain, with_offset), spacing


def check_interpolation_matches_np(solve):
    big = np.cumprod(np.linspace(0.9999, 0.98, 60))
    sigmas_full = np.sqrt((1.0 - big) / big)
    for steps in (2, 5, 13):
        for spacing in ("linspace", "leading", "trailing"):
            t, s = solve(torch.tensor(big, dtype=torch.float32), steps, timestep_spacing=spacing)
            want = np.interp(_grid_oracle(60, steps, spacing, 0), np.arange(0, 60), sigmas_full)
            assert np.allclose(s.numpy()[:-1], want.astype(np.float32), rtol=1e-6, atol=1e-6), (steps, spacing)
            # 整数时间点必须原样命中训练栅格上的 σ（trailing 浮点除法不整时会多出一个 -1 点，需排除）
            grid_t = t.numpy()
            exact = (grid_t >= 0) & np.isclose(grid_t, np.round(grid_t))
            if exact.any():
                picked = np.round(grid_t[exact]).astype(int)
                assert np.allclose(
                    s.numpy()[:-1][exact], sigmas_full[picked].astype(np.float32), rtol=1e-6, atol=1e-6
                ), (steps, spacing)


def check_terminal_sigma(solve):
    _, zero = solve(torch.tensor(ABAR, dtype=torch.float32), 4)
    assert float(zero[-1]) == 0.0
    _, smin = solve(torch.tensor(ABAR, dtype=torch.float32), 4, final_sigmas_type="sigma_min")
    want = math.sqrt((1.0 - ABAR[0]) / ABAR[0])
    assert abs(float(smin[-1]) - want) < 1e-6, (float(smin[-1]), want)
    # sigma_min 终端值就是训练栅格最干净那一端的 σ，因此会和倒数第二个点重合
    assert abs(float(smin[-1]) - float(smin[-2])) < 1e-6
    assert zero.shape[0] == smin.shape[0]


def check_karras(solve):
    _check(solve, steps=3, karras=True)
    _check(solve, steps=5, karras=True)
    _check(solve, steps=4, karras=True, final="sigma_min")

    abar = torch.tensor(ABAR, dtype=torch.float32)
    t, s = solve(abar, 5, use_karras_sigmas=True)
    sigmas_full = np.sqrt((1.0 - ABAR) / ABAR)
    assert abs(float(s[0]) - sigmas_full[-1]) < 1e-4, "Karras 栅格的首项应是噪声端 σ_max"
    assert abs(float(s[-2]) - sigmas_full[0]) < 1e-4, "Karras 栅格的末项应是干净端 σ_min"
    assert bool((s[1:] <= s[:-1] + 1e-7).all()), "σ 必须单调不增"

    # ρ=7 的显式公式：σ_i = (hi^(1/7) + ramp·(lo^(1/7) - hi^(1/7)))^7
    hi, lo = float(sigmas_full[-1]), float(sigmas_full[0])
    ramp = np.linspace(0.0, 1.0, 5)
    want = (hi ** (1 / 7.0) + ramp * (lo ** (1 / 7.0) - hi ** (1 / 7.0))) ** 7.0
    assert np.allclose(s.numpy()[:-1], want.astype(np.float32), rtol=1e-6, atol=1e-6), s.tolist()
    # ρ=7 的中段明显高于几何平均（ρ 越大越贴近噪声端），但仍低于 ρ=1 的线性插值 (hi+lo)/2
    geo = math.sqrt(hi * lo)
    assert float(s[2]) > geo
    assert float(s[2]) < 0.5 * (hi + lo)

    # 显式覆盖端点
    _, over = solve(abar, 3, use_karras_sigmas=True, sigma_min=0.5, sigma_max=20.0)
    assert abs(float(over[0]) - 20.0) < 1e-5
    assert abs(float(over[-2]) - 0.5) < 1e-6

    # 反查出来的时间必须落在训练栅格范围内且单调不增
    assert bool((t <= 9.0 + 1e-6).all()) and bool((t >= -1e-6).all())
    assert bool((t[1:] <= t[:-1] + 1e-5).all())


def check_exponential(solve):
    _check(solve, steps=3, exponential=True)
    _check(solve, steps=6, exponential=True)

    abar = torch.tensor(ABAR, dtype=torch.float32)
    _, s = solve(abar, 6, use_exponential_sigmas=True)
    logs = np.log(s.numpy()[:-1].astype(np.float64))
    gaps = np.diff(logs)
    assert np.allclose(gaps, gaps[0], atol=1e-5), "指数栅格的 log-σ 间隔应恒定"
    sigmas_full = np.sqrt((1.0 - ABAR) / ABAR)
    assert abs(float(s[0]) - sigmas_full[-1]) < 1e-4
    assert abs(float(s[-2]) - sigmas_full[0]) < 1e-4


def check_log_linear_quirk(solve):
    t, s = solve(torch.tensor(ABAR, dtype=torch.float32), 3, interpolation_type="log_linear")
    assert t.shape[0] == 3 and s.shape[0] == 5, (t.shape, s.shape)
    # log_linear 完全忽略时间栅格：σ 只由两端点与点数决定
    sigmas_full = np.sqrt((1.0 - ABAR) / ABAR)
    want = np.exp(np.linspace(math.log(sigmas_full[-1]), math.log(sigmas_full[0]), 4))
    assert np.allclose(s.numpy()[:-1], want.astype(np.float32), rtol=1e-6, atol=1e-6), s.tolist()
    shifted, s2 = solve(
        torch.tensor(ABAR, dtype=torch.float32), 3, timestep_spacing="leading", interpolation_type="log_linear"
    )
    assert torch.allclose(shifted, torch.tensor([6.0, 3.0, 0.0]))
    assert torch.equal(s, s2), "换 spacing 不改变 log_linear 的 σ 栅格"

    # 一旦挂上转换器，点数回到 N+1
    _, k = solve(torch.tensor(ABAR, dtype=torch.float32), 3, interpolation_type="log_linear", use_karras_sigmas=True)
    assert k.shape[0] == 4


def check_invalid(solve):
    abar = torch.tensor(ABAR, dtype=torch.float32)
    cases = [
        dict(alphas_cumprod=[0.9, 0.8]),
        dict(alphas_cumprod=torch.tensor([1, 0], dtype=torch.int64)),
        dict(alphas_cumprod=torch.tensor([[0.9, 0.8]])),
        dict(alphas_cumprod=torch.tensor([0.9])),
        dict(alphas_cumprod=torch.tensor([1.0, 0.9, 0.8])),
        dict(alphas_cumprod=torch.tensor([0.0, 0.5, 0.8])),
        dict(alphas_cumprod=torch.tensor([0.9, float("nan")])),
        dict(alphas_cumprod=abar, num_inference_steps=0),
        dict(alphas_cumprod=abar, num_inference_steps=11),
        dict(alphas_cumprod=abar, num_inference_steps=2.5),
        dict(alphas_cumprod=abar, num_inference_steps=True),
        dict(alphas_cumprod=abar, timestep_spacing="karras"),
        dict(alphas_cumprod=abar, interpolation_type="nearest"),
        dict(alphas_cumprod=abar, final_sigmas_type="sigma_max"),
        dict(alphas_cumprod=abar, use_karras_sigmas=True, use_exponential_sigmas=True),
        dict(alphas_cumprod=abar, use_karras_sigmas=True, use_beta_sigmas=True),
        dict(alphas_cumprod=abar, use_beta_sigmas=True),
        dict(alphas_cumprod=abar, steps_offset=-1),
        dict(alphas_cumprod=abar, sigma_min=0.0, use_karras_sigmas=True),
        dict(alphas_cumprod=abar, sigma_min=5.0, sigma_max=1.0, use_karras_sigmas=True),
    ]
    for kwargs in cases:
        steps = kwargs.pop("num_inference_steps", 3)
        alphas = kwargs.pop("alphas_cumprod")
        try:
            solve(alphas, steps, **kwargs)
            raise AssertionError(f"非法输入应抛 ValueError: {kwargs}")
        except ValueError:
            pass
