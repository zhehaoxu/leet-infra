import math

import numpy as np
import torch

ABAR = [0.99, 0.9, 0.7, 0.5, 0.3, 0.15, 0.06, 0.02, 0.005, 0.001]
TIMESTEPS = [9, 5, 2]
SAMPLE = np.array([[1.0, -2.0, 0.5], [0.25, 3.0, -1.5]], dtype=np.float64)
MODEL = np.array([[0.5, 0.25, -0.75], [0.1, -0.2, 0.3]], dtype=np.float64)
NOISE = np.array([[0.5, -0.5, 0.25], [-0.25, 0.75, 0.125]], dtype=np.float64)


def _oracle(
    sample,
    model_output,
    alphas,
    timesteps,
    step_index,
    prediction_type="epsilon",
    clip_sample=True,
    clip_sample_range=1.0,
    timestep_scaling=10.0,
    noise=None,
):
    x = np.asarray(sample, dtype=np.float64)
    model = np.asarray(model_output, dtype=np.float64)
    schedule = [int(v) for v in timesteps]
    timestep = schedule[step_index]
    prev_timestep = schedule[step_index + 1] if step_index + 1 < len(schedule) else timestep

    alpha_t = float(alphas[timestep])
    alpha_prev = float(alphas[prev_timestep]) if prev_timestep >= 0 else 1.0

    scaled = timestep * timestep_scaling
    sigma_data = 0.5
    c_skip = sigma_data**2 / (scaled**2 + sigma_data**2)
    c_out = scaled / math.sqrt(scaled**2 + sigma_data**2)

    if prediction_type == "epsilon":
        x0 = (x - math.sqrt(1.0 - alpha_t) * model) / math.sqrt(alpha_t)
    elif prediction_type == "sample":
        x0 = model.copy()
    else:
        x0 = math.sqrt(alpha_t) * x - math.sqrt(1.0 - alpha_t) * model
    if clip_sample:
        x0 = np.clip(x0, -clip_sample_range, clip_sample_range)

    denoised = c_out * x0 + c_skip * x
    if step_index != len(schedule) - 1:
        prev = math.sqrt(alpha_prev) * denoised + math.sqrt(max(1.0 - alpha_prev, 0.0)) * np.asarray(
            noise, dtype=np.float64
        )
    else:
        prev = denoised
    return denoised, prev, c_skip, c_out


def _run(solve, step_index=0, alphas=None, timesteps=None, sample=SAMPLE, model_output=MODEL, noise=NOISE, **kwargs):
    noise_arg = None if noise is None else torch.as_tensor(noise, dtype=torch.float32)
    return solve(
        torch.as_tensor(sample, dtype=torch.float32),
        torch.as_tensor(model_output, dtype=torch.float32),
        torch.as_tensor(ABAR if alphas is None else alphas, dtype=torch.float32),
        torch.tensor(TIMESTEPS if timesteps is None else timesteps),
        step_index,
        noise=noise_arg,
        **kwargs,
    )


def _check(solve, step_index=0, alphas=None, timesteps=None, sample=SAMPLE, model_output=MODEL, noise=NOISE, **kwargs):
    denoised, prev, c_skip, c_out = _run(
        solve, step_index, alphas, timesteps, sample, model_output, noise, **kwargs
    )
    abar = ABAR if alphas is None else alphas
    ts = TIMESTEPS if timesteps is None else timesteps
    rounded_x = np.asarray(sample, dtype=np.float32).astype(np.float64)
    rounded_model = np.asarray(model_output, dtype=np.float32).astype(np.float64)
    rounded_noise = None if noise is None else np.asarray(noise, dtype=np.float32).astype(np.float64)
    rounded_abar = [float(np.float32(v)) for v in abar]
    want_denoised, want_prev, want_skip, want_out = _oracle(
        rounded_x, rounded_model, rounded_abar, ts, step_index, noise=rounded_noise, **kwargs
    )
    for got, name in ((denoised, "denoised"), (prev, "prev")):
        assert got.dtype == torch.float32, (name, got.dtype)
    assert tuple(denoised.shape) == want_denoised.shape
    assert np.allclose(denoised.numpy(), want_denoised, rtol=1e-6, atol=1e-6), (denoised.tolist(), want_denoised.tolist())
    assert np.allclose(prev.numpy(), want_prev, rtol=1e-6, atol=1e-6), (prev.tolist(), want_prev.tolist())
    assert isinstance(c_skip, float) and isinstance(c_out, float)
    assert abs(c_skip - want_skip) < 1e-12 and abs(c_out - want_out) < 1e-12, (c_skip, c_out, want_skip, want_out)
    return denoised, prev, c_skip, c_out


def check_example(solve):
    denoised, prev, c_skip, c_out = _run(solve, step_index=0)
    assert abs(c_skip - 3.086324e-05) < 1e-9 and abs(c_out - 0.999985) < 1e-6, (c_skip, c_out)
    assert [[round(float(v), 6) for v in row] for row in denoised] == [
        [1.000015, -1.000046, 1.0],
        [0.999992, 1.000077, -1.000031],
    ], denoised.tolist()
    assert [[round(float(v), 6) for v in row] for row in prev] == [
        [0.848282, -0.848293, 0.617787],
        [0.156807, 1.078794, -0.272066],
    ], prev.tolist()

    denoised, prev, c_skip, c_out = _run(solve, step_index=2)
    assert abs(c_skip - 6.246096e-04) < 1e-9 and abs(c_out - 0.999688) < 1e-6, (c_skip, c_out)
    assert [[round(float(v), 6) for v in row] for row in denoised] == [
        [0.868255, -1.000937, 1.0],
        [0.233425, 1.001562, -1.000625],
    ], denoised.tolist()
    # 最后一步直接返回 denoised，不消耗 noise
    assert torch.equal(prev, denoised)

    _, _, c_skip_v, c_out_v = _run(solve, step_index=1, prediction_type="v_prediction")
    assert abs(c_skip_v - 9.999e-05) < 1e-9 and abs(c_out_v - 0.99995) < 1e-6


check_example.example = True


def check_skip_table(solve):
    _, _, c_skip, c_out = _run(solve, step_index=0)
    scaled = 90.0
    sigma_data = 0.5
    want_skip = sigma_data**2 / (scaled**2 + sigma_data**2)
    want_out = scaled / math.sqrt(scaled**2 + sigma_data**2)
    assert abs(c_skip - want_skip) < 1e-15 and abs(c_out - want_out) < 1e-15
    # LCM 的 c_out 少乘一个 σ_d：正好是 EDM 版本的两倍
    edm_out = scaled * sigma_data / math.sqrt(scaled**2 + sigma_data**2)
    assert abs(c_out - 2.0 * edm_out) < 1e-12
    # 因此 c_skip² + c_out² = c_skip 这条 EDM 恒等式在 LCM 下不成立
    assert abs(c_skip**2 + c_out**2 - c_skip) > 1e-3

    table = [_run(solve, timesteps=[t], step_index=0)[2:] for t in (0, 1, 4, 9)]
    # 随 t 增大：c_skip 单调下降、c_out 单调上升并趋近 1
    skips = [row[0] for row in table]
    outs = [row[1] for row in table]
    assert skips == sorted(skips, reverse=True), skips
    assert outs == sorted(outs), outs
    assert abs(table[0][0] - 1.0) < 1e-12 and table[0][1] == 0.0
    assert outs[-1] > 0.9999

    # timestep_scaling 把 t 整体拉长/压缩
    _, _, skip_half, out_half = _run(solve, timesteps=[9], step_index=0, timestep_scaling=5.0)
    assert abs(skip_half - 0.25 / (45.0**2 + 0.25)) < 1e-12
    assert abs(out_half - 45.0 / math.sqrt(45.0**2 + 0.25)) < 1e-12
    assert skip_half > c_skip and out_half < c_out

    # t=0 时 c_skip=1、c_out=0，denoised 退化成输入样本本身
    denoised, prev, c_skip0, c_out0 = _run(solve, timesteps=[0], step_index=0)
    assert c_skip0 == 1.0 and c_out0 == 0.0
    assert torch.allclose(denoised, torch.tensor(SAMPLE, dtype=torch.float32), atol=1e-6)
    assert torch.equal(prev, denoised)


def check_prediction_types(solve):
    abar = 0.3
    alphas = [0.9, 0.6, 0.45, abar]
    ts = [3, 2, 1]
    x = np.array([[0.8, -1.2], [0.4, 2.0]], dtype=np.float64)
    x0 = np.array([[0.3, -0.6], [0.5, 0.9]], dtype=np.float64)
    sqrt_a = math.sqrt(abar)
    sqrt_b = math.sqrt(1.0 - abar)
    eps = (x - sqrt_a * x0) / sqrt_b
    got, _, _, _ = _run(solve, step_index=0, alphas=alphas, timesteps=ts, sample=x, model_output=eps, noise=np.zeros_like(x))
    denoised, _, _, _ = _run(solve, step_index=0, alphas=alphas, timesteps=ts, sample=x, model_output=x0, noise=np.zeros_like(x), prediction_type="sample", clip_sample=False)
    assert np.allclose(got.numpy(), denoised.numpy(), atol=1e-6), (got.tolist(), denoised.tolist())

    v = sqrt_a * eps - sqrt_b * x0
    denoised, _, _, _ = _run(solve, step_index=0, alphas=alphas, timesteps=ts, sample=x, model_output=v, noise=np.zeros_like(x), prediction_type="v_prediction")
    assert np.allclose(got.numpy(), denoised.numpy(), atol=1e-6), (got.tolist(), denoised.tolist())

    # sample 支就是 c_out·model_output + c_skip·sample（无 clip 时）
    _, _, c_skip, c_out = _run(solve, step_index=0, alphas=alphas, timesteps=ts, sample=x, model_output=x0, noise=np.zeros_like(x), prediction_type="sample", clip_sample=False)
    want = c_out * x0 + c_skip * x
    assert np.allclose(denoised.numpy(), want, atol=1e-6), (denoised.tolist(), want.tolist())
    _check(solve, step_index=0, alphas=alphas, timesteps=ts, sample=x, model_output=v, noise=np.zeros_like(x), prediction_type="v_prediction")


def check_clipping(solve):
    x = np.array([[4.0, -6.0]], dtype=np.float64)
    model = np.array([[2.0, -3.0]], dtype=np.float64)
    ts = [9, 5, 2]
    clipped, _, _, _ = _run(solve, step_index=0, sample=x, model_output=model, noise=np.zeros_like(x), prediction_type="sample")
    raw, _, _, _ = _run(solve, step_index=0, sample=x, model_output=model, noise=np.zeros_like(x), prediction_type="sample", clip_sample=False)
    assert torch.allclose(clipped, torch.clamp(torch.tensor(model, dtype=torch.float32), -1.0, 1.0) * 1.0 + torch.tensor(x, dtype=torch.float32) * 3.086324e-05, atol=1e-5)
    assert not torch.allclose(clipped, raw)

    tight, _, _, _ = _run(solve, step_index=0, sample=x, model_output=model, noise=np.zeros_like(x), prediction_type="sample", clip_sample_range=0.4)
    assert float(tight.abs().max()) < 0.5, tight


def check_renoise(solve):
    abar = [0.9, 0.6, 0.3, 0.05]
    ts = [3, 2, 1]
    zero = _run(solve, step_index=0, alphas=abar, timesteps=ts, noise=np.zeros_like(SAMPLE))[1]
    one = _run(solve, step_index=0, alphas=abar, timesteps=ts, noise=np.ones_like(SAMPLE))[1]
    denoised = _run(solve, step_index=0, alphas=abar, timesteps=ts, noise=np.zeros_like(SAMPLE))[0]
    sqrt_prev = math.sqrt(abar[2])
    sqrt_beta = math.sqrt(1.0 - abar[2])
    assert np.allclose(zero.numpy(), sqrt_prev * denoised.numpy(), atol=1e-6)
    assert np.allclose((one - zero).numpy(), sqrt_beta, atol=1e-6), (one - zero).numpy()

    # 最后一步不消耗 noise
    a = _run(solve, step_index=2, alphas=abar, timesteps=ts, noise=np.zeros_like(SAMPLE))[1]
    b = _run(solve, step_index=2, alphas=abar, timesteps=ts, noise=np.full_like(SAMPLE, 1e3))[1]
    assert torch.equal(a, b)

    # 单步调度：唯一的这一步就是最后一步
    denoised, prev, _, _ = _run(solve, step_index=0, alphas=abar, timesteps=[2], noise=None)
    assert torch.equal(prev, denoised)


def check_prev_timestep_source(solve):
    abar = [0.9, 0.6, 0.3, 0.05]
    long_sched = _run(solve, step_index=0, alphas=abar, timesteps=[3, 2, 1], noise=np.zeros_like(SAMPLE))
    short_sched = _run(solve, step_index=0, alphas=abar, timesteps=[3, 1], noise=np.zeros_like(SAMPLE))
    # 当前 timestep 相同、denoised 相同，但下一步的 ᾱ 不同
    assert torch.allclose(long_sched[0], short_sched[0], atol=1e-6)
    assert not torch.allclose(long_sched[1], short_sched[1], atol=1e-4)

    # 允许重复 timestep：prev_timestep 取的是数组里的下一个元素，而不是 t-1
    repeated = _run(solve, step_index=0, alphas=abar, timesteps=[3, 3, 1], noise=np.zeros_like(SAMPLE))
    denoised = repeated[0]
    assert torch.allclose(repeated[1], math.sqrt(abar[3]) * denoised, atol=1e-6)
    assert not torch.allclose(repeated[1], math.sqrt(abar[2]) * denoised, atol=1e-4)

    _check(solve, step_index=1, alphas=abar, timesteps=[3, 3, 1], noise=NOISE)


def check_against_oracle(solve):
    rng = np.random.default_rng(23)
    schedules = ([9, 5, 2], [9, 9, 4, 0], [7, 3, 1], [8])
    for shape in ((2, 3), (2, 2, 3)):
        x = rng.normal(size=shape) * 1.5
        model = rng.normal(size=shape)
        noise = rng.normal(size=shape)
        for ts in schedules:
            for index in range(len(ts)):
                for kind in ("epsilon", "sample", "v_prediction"):
                    _check(solve, index, timesteps=ts, sample=x, model_output=model, noise=noise, prediction_type=kind)
                _check(solve, index, timesteps=ts, sample=x, model_output=model, noise=noise, clip_sample=False)
                _check(solve, index, timesteps=ts, sample=x, model_output=model, noise=noise, clip_sample_range=0.3)


def check_invalid(solve):
    x = torch.zeros((2, 3))
    model = torch.zeros((2, 3))
    abar = torch.tensor(ABAR)
    ts = torch.tensor(TIMESTEPS)
    noise = torch.zeros((2, 3))
    cases = [
        dict(sample=[[0.0, 0.0]], model_output=model, alphas_cumprod=abar, timesteps=ts, step_index=0, noise=noise),
        dict(sample=x.int(), model_output=model, alphas_cumprod=abar, timesteps=ts, step_index=0, noise=noise),
        dict(sample=x, model_output=torch.zeros((3, 3)), alphas_cumprod=abar, timesteps=ts, step_index=0, noise=noise),
        dict(sample=x, model_output=model, alphas_cumprod=ABAR, timesteps=ts, step_index=0, noise=noise),
        dict(sample=x, model_output=model, alphas_cumprod=torch.tensor([[0.9, 0.8]]), timesteps=ts, step_index=0, noise=noise),
        dict(sample=x, model_output=model, alphas_cumprod=torch.tensor([1.0, 0.9]), timesteps=ts, step_index=0, noise=noise),
        dict(sample=x, model_output=model, alphas_cumprod=torch.tensor([0.0, 0.9]), timesteps=ts, step_index=0, noise=noise),
        dict(sample=x, model_output=model, alphas_cumprod=abar, timesteps=[9, 5, 2], step_index=0, noise=noise),
        dict(sample=x, model_output=model, alphas_cumprod=abar, timesteps=torch.tensor([9.0, 5.0, 2.0]), step_index=0, noise=noise),
        dict(sample=x, model_output=model, alphas_cumprod=abar, timesteps=torch.tensor([9, 5, 10]), step_index=0, noise=noise),
        dict(sample=x, model_output=model, alphas_cumprod=abar, timesteps=torch.tensor([9, -1, 2]), step_index=0, noise=noise),
        dict(sample=x, model_output=model, alphas_cumprod=abar, timesteps=torch.tensor([[9, 5]]), step_index=0, noise=noise),
        dict(sample=x, model_output=model, alphas_cumprod=abar, timesteps=ts, step_index=-1, noise=noise),
        dict(sample=x, model_output=model, alphas_cumprod=abar, timesteps=ts, step_index=3, noise=noise),
        dict(sample=x, model_output=model, alphas_cumprod=abar, timesteps=ts, step_index=True, noise=noise),
        dict(sample=x, model_output=model, alphas_cumprod=abar, timesteps=ts, step_index=0, noise=noise, prediction_type="flow"),
        dict(sample=x, model_output=model, alphas_cumprod=abar, timesteps=ts, step_index=0, noise=noise, clip_sample=1),
        dict(sample=x, model_output=model, alphas_cumprod=abar, timesteps=ts, step_index=0, noise=noise, clip_sample_range=0.0),
        dict(sample=x, model_output=model, alphas_cumprod=abar, timesteps=ts, step_index=0, noise=noise, timestep_scaling=-1.0),
        dict(sample=x, model_output=model, alphas_cumprod=abar, timesteps=ts, step_index=0, noise=None),
        dict(sample=x, model_output=model, alphas_cumprod=abar, timesteps=ts, step_index=0, noise=torch.zeros((1, 3))),
    ]
    for kwargs in cases:
        try:
            solve(kwargs.pop("sample"), kwargs.pop("model_output"), kwargs.pop("alphas_cumprod"), kwargs.pop("timesteps"), **kwargs)
            raise AssertionError(f"非法输入应抛 ValueError: {kwargs}")
        except ValueError:
            pass
