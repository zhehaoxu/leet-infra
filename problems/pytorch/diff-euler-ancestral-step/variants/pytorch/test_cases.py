import math

import numpy as np
import torch

SAMPLE = np.array([[1.0, -2.0, 0.5], [0.25, 3.0, -1.5]], dtype=np.float64)
MODEL = np.array([[0.5, 0.25, -0.75], [0.1, -0.2, 0.3]], dtype=np.float64)
NOISE = np.array([[0.5, -0.5, 0.25], [-0.25, 0.75, 0.125]], dtype=np.float64)
SIGMAS = [2.0, 1.0, 0.5, 0.0]


def _oracle(sample, model_output, sigmas, step_index, prediction_type="epsilon", noise=None):
    x = np.asarray(sample, dtype=np.float64)
    model = np.asarray(model_output, dtype=np.float64)
    sigma_from = float(sigmas[step_index])
    sigma_to = float(sigmas[step_index + 1])
    sigma_up = math.sqrt(sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2)
    sigma_down = math.sqrt(max(sigma_to**2 - sigma_up**2, 0.0))
    scaled = x / math.sqrt(sigma_from**2 + 1.0)
    if prediction_type == "epsilon":
        x0 = x - sigma_from * model
    else:
        x0 = model * (-sigma_from / math.sqrt(sigma_from**2 + 1.0)) + x / (sigma_from**2 + 1.0)
    prev = x + (x - x0) / sigma_from * (sigma_down - sigma_from)
    if sigma_up > 0.0:
        prev = prev + np.asarray(noise, dtype=np.float64) * sigma_up
    return scaled, prev, x0, sigma_up, sigma_down


def _run(solve, sigmas=None, sample=SAMPLE, model_output=MODEL, step_index=0, noise=NOISE, **kwargs):
    if noise is None:
        noise_arg = None
    else:
        noise_arg = torch.as_tensor(noise, dtype=torch.float32)
    return solve(
        torch.as_tensor(sample, dtype=torch.float32),
        torch.as_tensor(model_output, dtype=torch.float32),
        torch.as_tensor(SIGMAS if sigmas is None else sigmas, dtype=torch.float32),
        step_index,
        noise=noise_arg,
        **kwargs,
    )


def _check(solve, sigmas=None, sample=SAMPLE, model_output=MODEL, step_index=0, noise=NOISE, **kwargs):
    scaled, prev, x0, up, down = _run(solve, sigmas, sample, model_output, step_index, noise, **kwargs)
    # 参考解先被压到 float32，oracle 必须吃同一批舍入后的输入
    rounded_sample = np.asarray(sample, dtype=np.float32).astype(np.float64)
    rounded_model = np.asarray(model_output, dtype=np.float32).astype(np.float64)
    rounded_noise = None if noise is None else np.asarray(noise, dtype=np.float32).astype(np.float64)
    rounded_sigmas = [float(np.float32(v)) for v in (SIGMAS if sigmas is None else sigmas)]
    want_scaled, want_prev, want_x0, want_up, want_down = _oracle(
        rounded_sample, rounded_model, rounded_sigmas, step_index, noise=rounded_noise, **kwargs
    )
    for got, want, name in ((scaled, want_scaled, "scaled"), (prev, want_prev, "prev"), (x0, want_x0, "x0")):
        assert got.dtype == torch.float32, (name, got.dtype)
        assert tuple(got.shape) == want.shape, (name, got.shape, want.shape)
        assert np.allclose(got.numpy(), want, rtol=1e-6, atol=1e-6), (name, got.tolist(), want.tolist())
    assert isinstance(up, float) and isinstance(down, float)
    assert abs(up - want_up) < 1e-6 and abs(down - want_down) < 1e-6, (up, down, want_up, want_down)
    return scaled, prev, x0, up, down


def check_example(solve):
    scaled, prev, x0, up, down = _run(solve, step_index=0)
    assert abs(up - 0.866025) < 1e-6 and abs(down - 0.5) < 1e-6, (up, down)
    assert [[round(float(v), 6) for v in row] for row in prev] == [
        [0.683013, -2.808013, 1.841506],
        [-0.116506, 3.949519, -1.841747],
    ], prev.tolist()
    assert [[round(float(v), 6) for v in row] for row in x0] == [
        [0.0, -2.5, 2.0],
        [0.05, 3.4, -2.1],
    ], x0.tolist()
    assert [[round(float(v), 6) for v in row] for row in scaled] == [
        [0.447214, -0.894427, 0.223607],
        [0.111803, 1.341641, -0.67082],
    ], scaled.tolist()

    _, prev, _, up, down = _run(solve, step_index=1)
    assert abs(up - 0.433013) < 1e-6 and abs(down - 0.25) < 1e-6, (up, down)
    assert [[round(float(v), 6) for v in row] for row in prev] == [
        [0.841506, -2.404006, 1.170753],
        [0.066747, 3.47476, -1.670873],
    ], prev.tolist()

    _, prev, x0, up, down = _run(solve, step_index=2)
    assert up == 0.0 and down == 0.0, (up, down)
    assert torch.equal(prev, x0), (prev.tolist(), x0.tolist())


check_example.example = True


def check_coefficient_identities(solve):
    sigmas = [4.0, 2.5, 1.0, 0.3, 0.0]
    for index in range(len(sigmas) - 1):
        _, _, _, up, down = _run(solve, sigmas=sigmas, step_index=index)
        sigma_from = sigmas[index]
        sigma_to = sigmas[index + 1]
        assert abs(up**2 + down**2 - sigma_to**2) < 1e-6, (index, up, down, sigma_to)
        # sigma_up 就是「把 sigma_to 的一部分挪到噪声项」：1 - sigma_to²/sigma_from² 的比例
        assert abs(up**2 - sigma_to**2 * (1.0 - (sigma_to / sigma_from) ** 2)) < 1e-6, index
        assert 0.0 <= up <= sigma_to + 1e-12 and down >= 0.0, (index, up, down)

    # sigma_to -> sigma_from 时随机项消失、确定项吃满整个下降量
    _, _, _, up, down = _run(solve, sigmas=[2.0, 1.999999, 0.0], step_index=0)
    assert up < 2e-3 and abs(down - 1.999999) < 1e-5, (up, down)

    # sigma_to -> 0 时两项都趋于 0
    _, _, _, up, down = _run(solve, sigmas=[2.0, 1e-6, 0.0], step_index=0)
    assert up < 1e-5 and down < 1e-5, (up, down)


def check_last_step(solve):
    # 终端 sigma = 0：sigma_up = sigma_down = 0，prev 精确等于 x0，且不需要 noise
    for index in (2,):
        scaled, prev, x0, up, down = _run(solve, step_index=index, noise=None)
        assert up == 0.0 and down == 0.0
        assert torch.equal(prev, x0), (prev.tolist(), x0.tolist())

    # 传入完全不同的 noise 也不影响最后一步
    _, a, x0, _, _ = _run(solve, step_index=2, noise=np.zeros_like(SAMPLE))
    _, b, _, _, _ = _run(solve, step_index=2, noise=np.full_like(SAMPLE, 1e6))
    assert torch.equal(a, b) and torch.equal(a, x0)

    # σ_to = σ_from（原地不动）：dt = 0 且 sigma_up = 0
    flat = [1.5, 1.5, 0.0]
    _, prev, _, up, down = _run(solve, sigmas=flat, step_index=0, noise=None)
    assert up == 0.0 and abs(down - 1.5) < 1e-9
    assert torch.equal(prev, torch.tensor(SAMPLE, dtype=torch.float32))


def check_noise_scale(solve):
    base_scaled, base_prev, x0, up, down = _run(solve, step_index=1, noise=np.zeros_like(SAMPLE))
    bumped_scaled, bumped_prev, _, _, _ = _run(solve, step_index=1)
    assert torch.equal(base_scaled, bumped_scaled)
    delta = (bumped_prev - base_prev).numpy()
    assert np.allclose(delta, NOISE * up, atol=1e-6), (delta.tolist(), (NOISE * up).tolist())
    # noise 只按 sigma_up 缩放，与 sigma_down 无关
    assert abs(up - 0.433013) < 1e-6


def check_v_prediction(solve):
    sigma = 1.0
    sigmas = [sigma, 0.5, 0.0]
    _, _, x0, _, _ = _run(solve, sigmas=sigmas, step_index=0, prediction_type="v_prediction")
    x = SAMPLE
    model = MODEL
    alpha = 1.0 / math.sqrt(sigma**2 + 1.0)
    sigma_t = sigma * alpha
    scaled = x / math.sqrt(sigma**2 + 1.0)
    # 上游写法：x0 = -σ_t·v + α·scaled，等价于样本项少乘一个 α
    upstream = -sigma_t * model + alpha * scaled
    assert np.allclose(x0.numpy(), upstream, atol=1e-6), (x0.tolist(), upstream.tolist())
    textbook = alpha * x - sigma_t * model
    assert not np.allclose(x0.numpy(), textbook, atol=1e-3), "上游分支与教科书公式不同，不应相等"
    # 小 σ 下两者的差异按 α(1-α) 收缩：σ → 0 时上游分支收敛到教科书公式
    gaps = []
    for small in (1.0, 0.5, 0.1, 0.01):
        _, _, got, _, _ = _run(solve, sigmas=[small, small / 2, 0.0], step_index=0, prediction_type="v_prediction")
        a = 1.0 / math.sqrt(small**2 + 1.0)
        st = small * a
        textbook_small = a * x - st * model
        gaps.append(float(np.abs(got.numpy() - textbook_small).max()))
    for before, after in zip(gaps, gaps[1:]):
        assert after < before, gaps

    _check(solve, step_index=0, prediction_type="v_prediction")


def check_epsilon_roundtrip(solve):
    sigma = 2.0
    x0_true = np.array([[0.3, -0.4], [0.9, 0.1]], dtype=np.float64)
    x = np.array([[1.5, -2.5], [0.5, 2.0]], dtype=np.float64)
    eps = (x - x0_true) / sigma
    scaled, prev, x0, up, down = _run(solve, sigmas=[sigma, 1.0, 0.0], sample=x, model_output=eps, noise=np.zeros_like(x))
    assert np.allclose(x0.numpy(), x0_true, atol=1e-6), (x0.tolist(), x0_true.tolist())
    want_prev = x + (x - x0_true) / sigma * (down - sigma)
    assert np.allclose(prev.numpy(), want_prev, atol=1e-6), (prev.tolist(), want_prev.tolist())
    assert np.allclose(scaled.numpy(), x / math.sqrt(sigma**2 + 1.0), atol=1e-6)


def check_against_oracle(solve):
    rng = np.random.default_rng(11)
    schedules = [
        [3.0, 1.5, 0.0],
        [2.0, 1.999999, 0.5, 0.0],
        [5.0, 2.0, 0.7, 0.05, 0.0],
        [1.0, 0.5, 0.25, 0.1, 0.02, 0.0],
    ]
    for shape in ((2, 3), (2, 2, 3)):
        x = rng.normal(size=shape) * 2.0
        model = rng.normal(size=shape)
        noise = rng.normal(size=shape)
        for sigmas in schedules:
            for index in range(len(sigmas) - 1):
                for kind in ("epsilon", "v_prediction"):
                    _check(solve, sigmas, x, model, index, noise, prediction_type=kind)


def check_invalid(solve):
    x = torch.zeros((2, 3))
    model = torch.zeros((2, 3))
    sig = torch.tensor(SIGMAS)
    noise = torch.zeros((2, 3))
    cases = [
        dict(sample=[[0.0, 0.0]], model_output=model, sigmas=sig, step_index=0, noise=noise),
        dict(sample=x.double(), model_output=model.int(), sigmas=sig, step_index=0, noise=noise),
        dict(sample=x, model_output=torch.zeros((3, 3)), sigmas=sig, step_index=0, noise=noise),
        dict(sample=x, model_output=model, sigmas=[2.0, 1.0, 0.0], step_index=0, noise=noise),
        dict(sample=x, model_output=model, sigmas=torch.tensor([[2.0, 1.0]]), step_index=0, noise=noise),
        dict(sample=x, model_output=model, sigmas=torch.tensor([1.0]), step_index=0, noise=noise),
        dict(sample=x, model_output=model, sigmas=torch.tensor([1.0, 2.0, 0.0]), step_index=0, noise=noise),
        dict(sample=x, model_output=model, sigmas=torch.tensor([1.0, -1.0, 0.0]), step_index=0, noise=noise),
        dict(sample=x, model_output=model, sigmas=torch.tensor([1.0, float("inf")]), step_index=0, noise=noise),
        dict(sample=x, model_output=model, sigmas=sig, step_index=-1, noise=noise),
        dict(sample=x, model_output=model, sigmas=sig, step_index=3, noise=noise),
        dict(sample=x, model_output=model, sigmas=sig, step_index=True, noise=noise),
        dict(sample=x, model_output=model, sigmas=sig, step_index=1.0, noise=noise),
        dict(sample=x, model_output=model, sigmas=sig, step_index=0, noise=noise, prediction_type="sample"),
        dict(sample=x, model_output=model, sigmas=sig, step_index=0, noise=noise, prediction_type="flow"),
        dict(sample=x, model_output=model, sigmas=sig, step_index=0, noise=None),
        dict(sample=x, model_output=model, sigmas=sig, step_index=0, noise=torch.zeros((1, 3))),
        dict(sample=x, model_output=model, sigmas=torch.tensor([0.0, 0.0, 0.0]), step_index=0, noise=noise),
    ]
    for kwargs in cases:
        try:
            solve(kwargs.pop("sample"), kwargs.pop("model_output"), kwargs.pop("sigmas"), **kwargs)
            raise AssertionError(f"非法输入应抛 ValueError: {kwargs}")
        except ValueError:
            pass
