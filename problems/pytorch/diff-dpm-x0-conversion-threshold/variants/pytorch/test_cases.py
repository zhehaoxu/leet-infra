import numpy as np
import torch

SAMPLE = np.array([[1.0, -2.0, 0.5], [0.25, 3.0, -1.5]], dtype=np.float64)
EPS = np.array([[0.5, 0.25, -0.75], [0.1, -0.2, 0.3]], dtype=np.float64)


def _threshold_oracle(x, ratio, max_value):
    batch = x.shape[0]
    flat = np.abs(x.reshape(batch, -1))
    scale = np.quantile(flat, ratio, axis=1, method="linear")
    scale = np.clip(scale, 1.0, max_value)[:, None]
    return (np.clip(x.reshape(batch, -1), -scale, scale) / scale).reshape(x.shape)


def _oracle(
    model_output,
    sample,
    sigma,
    prediction_type="epsilon",
    algorithm_type="dpmsolver++",
    variance_type="fixed_small",
    thresholding=False,
    dynamic_thresholding_ratio=0.995,
    sample_max_value=1.0,
    use_flow_sigmas=False,
):
    mo = np.asarray(model_output, dtype=np.float64)
    x = np.asarray(sample, dtype=np.float64)
    if variance_type in ("learned", "learned_range"):
        mo = mo[:, :3]
    if use_flow_sigmas:
        alpha = 1.0 - sigma
        sigma_t = sigma
    else:
        alpha = 1.0 / np.sqrt(sigma * sigma + 1.0)
        sigma_t = sigma * alpha

    if algorithm_type in ("dpmsolver++", "sde-dpmsolver++"):
        if prediction_type == "epsilon":
            out = (x - sigma_t * mo) / alpha
        elif prediction_type == "sample":
            out = mo.copy()
        elif prediction_type == "v_prediction":
            out = alpha * x - sigma_t * mo
        else:
            out = x - sigma * mo
        if thresholding:
            out = _threshold_oracle(out, dynamic_thresholding_ratio, sample_max_value)
    else:
        if prediction_type == "epsilon":
            eps = mo.copy()
        elif prediction_type == "sample":
            eps = (x - alpha * mo) / sigma_t
        else:
            eps = alpha * mo + sigma_t * x
        if thresholding:
            x0 = _threshold_oracle(
                (x - sigma_t * eps) / alpha, dynamic_thresholding_ratio, sample_max_value
            )
            eps = (x - alpha * x0) / sigma_t
        out = eps
    return out.astype(np.float32)


def _run(solve, model_output, sample, sigma, **kwargs):
    return solve(
        torch.as_tensor(model_output, dtype=torch.float32),
        torch.as_tensor(sample, dtype=torch.float32),
        sigma,
        **kwargs,
    )


def _check(solve, model_output, sample, sigma, **kwargs):
    got = _run(solve, model_output, sample, sigma, **kwargs)
    want = _oracle(model_output, sample, sigma, **kwargs)
    assert got.dtype == torch.float32, got.dtype
    assert tuple(got.shape) == np.asarray(sample).shape, (got.shape, np.asarray(sample).shape)
    assert np.allclose(got.numpy(), want, rtol=1e-6, atol=1e-5), (got.tolist(), want.tolist())
    return got


def check_example(solve):
    r = _run(solve, EPS, SAMPLE, 0.5)
    assert [[round(float(v), 6) for v in row] for row in r] == [
        [0.868034, -2.361068, 0.934017],
        [0.229508, 3.454102, -1.827051],
    ], r.tolist()

    r = _run(solve, EPS, SAMPLE, 0.5, prediction_type="v_prediction")
    assert [[round(float(v), 6) for v in row] for row in r] == [
        [0.67082, -1.900658, 0.782624],
        [0.178885, 2.772724, -1.475805],
    ], r.tolist()

    # 噪声口径 + epsilon：原样返回
    plain = _run(solve, EPS, SAMPLE, 0.5, algorithm_type="dpmsolver")
    assert torch.equal(plain, torch.tensor(EPS, dtype=torch.float32))

    r = _run(solve, EPS, SAMPLE, 0.5, algorithm_type="dpmsolver", prediction_type="v_prediction")
    assert [[round(float(v), 6) for v in row] for row in r] == [
        [0.894427, -0.67082, -0.447214],
        [0.201246, 1.162755, -0.402492],
    ], r.tolist()

    r = _run(
        solve,
        EPS,
        SAMPLE,
        0.5,
        thresholding=True,
        dynamic_thresholding_ratio=0.9,
        sample_max_value=1.2,
    )
    assert [[round(float(v), 6) for v in row] for row in r] == [
        [0.723362, -1.0, 0.778347],
        [0.191257, 1.0, -1.0],
    ], r.tolist()

    # σ=0.5 时 α=1/√1.25=0.894427、σ_t=0.447214；0.9 与 1.0 两个分位数都被 1.2 截断，
    # 因此两者结果相同——这是 sample_max_value 上限钳位的直接后果
    same = _run(
        solve,
        EPS,
        SAMPLE,
        0.5,
        thresholding=True,
        dynamic_thresholding_ratio=1.0,
        sample_max_value=1.2,
    )
    assert torch.equal(same, r)


check_example.example = True


def check_prediction_types(solve):
    x = np.array([[0.4, -0.9], [1.3, 0.05]], dtype=np.float64)
    sigma = 0.7
    alpha = 1.0 / np.sqrt(sigma**2 + 1.0)
    sigma_t = sigma * alpha
    x0 = np.array([[0.2, -0.5], [0.9, -0.1]], dtype=np.float64)

    eps = (x - alpha * x0) / sigma_t
    got = _run(solve, eps, x, sigma)
    assert np.allclose(got.numpy(), x0, atol=1e-5), (got.tolist(), x0)
    _check(solve, eps, x, sigma)

    v = alpha * eps - sigma_t * x0
    got = _run(solve, v, x, sigma, prediction_type="v_prediction")
    assert np.allclose(got.numpy(), x0, atol=1e-5), (got.tolist(), x0)

    # sample 预测直接返回网络输出（阈值关闭时），并与 oracle 一致
    got = _run(solve, x0, x, sigma, prediction_type="sample")
    assert torch.equal(got, torch.tensor(x0, dtype=torch.float32))
    _check(solve, x0, x, sigma, prediction_type="sample")

    # flow_prediction 的 model_output 是速度 v = ε - x0，且用原始 σ 而不是 alpha/sigma_t
    flow_x0 = np.array([[0.1, 0.3], [-0.7, 0.6]], dtype=np.float64)
    flow_sigma = 0.4
    flow_v = eps - flow_x0
    flow_x = (1.0 - flow_sigma) * flow_x0 + flow_sigma * eps
    got = _run(solve, flow_v, flow_x, flow_sigma, prediction_type="flow_prediction", use_flow_sigmas=True)
    assert np.allclose(got.numpy(), flow_x0, atol=1e-5), (got.tolist(), flow_x0)
    _check(solve, flow_v, flow_x, flow_sigma, prediction_type="flow_prediction", use_flow_sigmas=True)


def check_noise_branch(solve):
    x = np.array([[0.6, -1.4], [0.2, 2.2]], dtype=np.float64)
    sigma = 0.9
    alpha = 1.0 / np.sqrt(sigma**2 + 1.0)
    sigma_t = sigma * alpha
    eps = np.array([[0.1, -0.3], [0.7, 0.4]], dtype=np.float64)

    got = _run(solve, eps, x, sigma, algorithm_type="sde-dpmsolver")
    assert torch.equal(got, torch.tensor(eps, dtype=torch.float32))

    x0 = (x - sigma_t * eps) / alpha
    got = _run(solve, x0, x, sigma, algorithm_type="dpmsolver", prediction_type="sample")
    assert np.allclose(got.numpy(), eps, atol=1e-5), (got.tolist(), eps)

    v = alpha * eps - sigma_t * x0
    got = _run(solve, v, x, sigma, algorithm_type="dpmsolver", prediction_type="v_prediction")
    assert np.allclose(got.numpy(), eps, atol=1e-5), (got.tolist(), eps)

    # 噪声口径 + sample 预测（不阈值）与数据口径 + epsilon 互逆
    back = _run(solve, eps, x, sigma)
    assert np.allclose(back.numpy(), x0, atol=1e-5), (back.tolist(), x0)


def check_thresholding_math(solve):
    inside = np.array([[0.3, -0.8], [0.99, -0.95]], dtype=np.float64)
    got = _run(solve, inside, inside, 0.5, prediction_type="sample", thresholding=True)
    # 分位数 < 1 时 s 被 min=1 托住，样本已在 [-1, 1] 内，输出应逐位不变
    assert torch.equal(got, torch.tensor(inside, dtype=torch.float32))

    out = _run(solve, inside * 4.0, inside * 4.0, 0.5, prediction_type="sample", thresholding=True)
    assert float(out.abs().max()) <= 1.0 + 1e-6, out.abs().max()

    wide = np.array([[3.0, -6.0], [1.5, 9.0]], dtype=np.float64)
    got = _run(
        solve,
        wide,
        wide,
        0.5,
        prediction_type="sample",
        thresholding=True,
        dynamic_thresholding_ratio=1.0,
        sample_max_value=100.0,
    )
    want = wide / np.abs(wide).max(axis=1, keepdims=True)
    assert np.allclose(got.numpy(), want, atol=1e-5), (got.tolist(), want.tolist())

    # 每个 batch 元素独立取分位数：改一个样本不影响另一个
    a = np.array([[0.2, 0.4], [5.0, -7.0]], dtype=np.float64)
    b = a.copy()
    b[0] = [0.9, -0.9]
    got_a = _run(solve, a, a, 0.5, prediction_type="sample", thresholding=True)
    got_b = _run(solve, b, b, 0.5, prediction_type="sample", thresholding=True)
    assert torch.equal(got_a[1], got_b[1])

    # s 的上限由 sample_max_value 截断：阈值化后的绝对值永远落在 [0, 1]
    huge = np.array([[50.0, -80.0]], dtype=np.float64)
    tight = _run(
        solve,
        huge,
        huge,
        0.5,
        prediction_type="sample",
        thresholding=True,
        dynamic_thresholding_ratio=1.0,
        sample_max_value=1.0,
    )
    assert np.allclose(tight.numpy(), [[1.0, -1.0]], atol=1e-6), tight.tolist()
    loose = _run(
        solve,
        huge,
        huge,
        0.5,
        prediction_type="sample",
        thresholding=True,
        dynamic_thresholding_ratio=1.0,
        sample_max_value=100.0,
    )
    assert np.allclose(loose.numpy(), [[0.625, -1.0]], atol=1e-6), loose.tolist()


def check_flow_sigmas(solve):
    x = np.array([[0.2, -0.4, 0.6], [-0.1, 0.35, 0.8]], dtype=np.float64)
    sigma = 0.6
    alpha = 1.0 - sigma
    sigma_t = sigma
    x0 = np.array([[0.0, 0.5, -0.25], [0.3, -0.6, 0.15]], dtype=np.float64)
    eps = (x - alpha * x0) / sigma_t
    _check(solve, eps, x, sigma, use_flow_sigmas=True)

    got = _run(solve, eps, x, sigma, use_flow_sigmas=True)
    assert np.allclose(got.numpy(), x0, atol=1e-5), (got.tolist(), x0)

    # 同一组输入下，flow 口径与 VP 口径的 α 不同，结果必然不同
    vp = _run(solve, eps, x, sigma)
    assert not torch.allclose(vp, got, atol=1e-3)

    noise = _run(solve, eps, x, sigma, use_flow_sigmas=True, algorithm_type="dpmsolver")
    assert torch.equal(noise, torch.tensor(eps, dtype=torch.float32))


def check_learned_variance(solve):
    mo6 = np.concatenate([EPS, np.full((2, 3), 9.0)], axis=1)
    got = _run(solve, mo6, SAMPLE, 0.5, variance_type="learned")
    want = _run(solve, EPS, SAMPLE, 0.5)
    assert torch.equal(got, want)

    got = _run(solve, mo6, SAMPLE, 0.5, variance_type="learned_range", algorithm_type="dpmsolver")
    want = _run(solve, EPS, SAMPLE, 0.5, algorithm_type="dpmsolver")
    assert torch.equal(got, want)
    _check(solve, mo6, SAMPLE, 0.5, variance_type="learned", thresholding=True)


def check_against_oracle(solve):
    rng = np.random.default_rng(7)
    shapes = [(2, 3), (3, 2, 4)]
    configs = [
        dict(),
        dict(prediction_type="sample"),
        dict(prediction_type="v_prediction"),
        dict(prediction_type="flow_prediction", use_flow_sigmas=True),
        dict(algorithm_type="dpmsolver"),
        dict(algorithm_type="sde-dpmsolver", prediction_type="v_prediction"),
        dict(algorithm_type="sde-dpmsolver", prediction_type="sample"),
        dict(thresholding=True),
        dict(thresholding=True, dynamic_thresholding_ratio=0.9),
        dict(thresholding=True, sample_max_value=1.5, dynamic_thresholding_ratio=1.0),
        dict(algorithm_type="dpmsolver", thresholding=True),
        dict(algorithm_type="dpmsolver", prediction_type="sample", thresholding=True),
        dict(prediction_type="flow_prediction", use_flow_sigmas=True, thresholding=True),
        dict(use_flow_sigmas=True),
    ]
    for shape in shapes:
        for sigma in (0.05, 0.4, 1.5):
            x = rng.normal(size=shape) * 1.4
            mo = rng.normal(size=shape) * 0.6
            for cfg in configs:
                if cfg.get("use_flow_sigmas") and sigma >= 1.0:
                    continue
                _check(solve, mo, x, sigma, **cfg)

    mo6 = rng.normal(size=(2, 6, 3))
    x3 = rng.normal(size=(2, 3, 3))
    for cfg in (dict(variance_type="learned"), dict(variance_type="learned_range", thresholding=True)):
        _check(solve, mo6, x3, 0.8, **cfg)


def check_invalid(solve):
    mo = np.zeros((2, 3))
    x = np.zeros((2, 3))
    cases = [
        dict(model_output=[[0.0, 0.0, 0.0]], sample=x, sigma=0.5),
        dict(model_output=torch.tensor([[0, 0, 0]], dtype=torch.int64), sample=x, sigma=0.5),
        dict(model_output=mo, sample=x, sigma=0.0),
        dict(model_output=mo, sample=x, sigma=-0.5),
        dict(model_output=mo, sample=x, sigma=float("nan")),
        dict(model_output=mo, sample=x, sigma=True),
        dict(model_output=mo, sample=x, sigma=0.5, prediction_type="velocity"),
        dict(model_output=mo, sample=x, sigma=0.5, algorithm_type="deis"),
        dict(model_output=mo, sample=x, sigma=0.5, variance_type="fixed"),
        dict(model_output=mo, sample=x, sigma=0.5, thresholding=1),
        dict(model_output=mo, sample=x, sigma=0.5, use_flow_sigmas="yes"),
        dict(model_output=mo, sample=x, sigma=0.5, dynamic_thresholding_ratio=0.0),
        dict(model_output=mo, sample=x, sigma=0.5, dynamic_thresholding_ratio=1.5),
        dict(model_output=mo, sample=x, sigma=0.5, sample_max_value=0.5),
        dict(model_output=mo, sample=x, sigma=1.0, use_flow_sigmas=True),
        dict(model_output=mo, sample=x, sigma=0.5, algorithm_type="dpmsolver", prediction_type="flow_prediction"),
        dict(model_output=mo, sample=x, sigma=0.5, variance_type="learned"),
        dict(model_output=np.zeros((2, 6, 3)), sample=np.zeros((2, 3, 3)), sigma=0.5, variance_type="learned", prediction_type="sample"),
        dict(model_output=np.zeros((2, 3, 4)), sample=np.zeros((2, 3, 3)), sigma=0.5),
        dict(model_output=np.zeros((3,)), sample=np.zeros((3,)), sigma=0.5),
        dict(model_output=np.zeros((0, 3)), sample=np.zeros((0, 3)), sigma=0.5),
    ]
    for kwargs in cases:
        try:
            _run(solve, kwargs.pop("model_output"), kwargs.pop("sample"), **kwargs)
            raise AssertionError(f"非法输入应抛 ValueError: {kwargs}")
        except ValueError:
            pass
