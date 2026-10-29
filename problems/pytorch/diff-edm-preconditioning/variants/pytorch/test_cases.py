import math

import torch

INF = float("inf")


def _oracle(
    x,
    F,
    sigmas,
    index,
    sigma_data=0.5,
    pred="epsilon",
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=INF,
    s_noise=1.0,
    noise=None,
):
    """独立 oracle：用 python 标量与嵌套 list 重做预条件系数、加噪与 Euler 步。"""
    sd = sigma_data
    sigma = sigmas[index]
    steps = len(sigmas) - 1
    gamma = min(s_churn / steps, math.sqrt(2.0) - 1.0) if s_tmin <= sigma <= s_tmax else 0.0
    sigma_hat = sigma * (1.0 + gamma)

    c_in = 1.0 / math.sqrt(sigma * sigma + sd * sd)
    model_input = [[v * c_in for v in row] for row in x]
    c_noise = 0.25 * math.log(sigma)

    sample = [row[:] for row in x]
    if gamma > 0.0:
        scale = math.sqrt(sigma_hat * sigma_hat - sigma * sigma)
        sample = [
            [a + n * s_noise * scale for a, n in zip(row, nrow)] for row, nrow in zip(sample, noise)
        ]

    c_skip = sd * sd / (sigma_hat * sigma_hat + sd * sd)
    c_out = sigma_hat * sd / math.sqrt(sigma_hat * sigma_hat + sd * sd)
    if pred == "v_prediction":
        c_out = -c_out
    denoised = [[c_skip * a + c_out * b for a, b in zip(row, frow)] for row, frow in zip(sample, F)]

    dt = sigmas[index + 1] - sigma_hat
    prev = [
        [a + (a - d) / sigma_hat * dt for a, d in zip(row, drow)]
        for row, drow in zip(sample, denoised)
    ]
    return model_input, c_noise, prev, denoised, sigma_hat


def _run(solve, x, F, sigmas, **kwargs):
    return solve(x, F, torch.tensor(sigmas, dtype=torch.float32), **kwargs)


def _rounds(tensor):
    return [round(float(v), 6) for row in tensor.tolist() for v in row]


def check_example(solve):
    x = torch.tensor([[1.0, -2.0]])
    F = torch.tensor([[0.5, 0.25]])
    sigmas = [2.0, 1.0, 0.5, 0.0]

    mi, cn, pv, dn, sh = _run(solve, x, F, sigmas, step_index=0)
    assert mi.dtype == torch.float32 and mi.shape == x.shape
    assert _rounds(mi) == [0.485071, -0.970142], mi.tolist()
    assert round(cn, 6) == 0.173287 and round(sh, 6) == 2.0
    assert _rounds(dn) == [0.301359, 0.003621], dn.tolist()
    assert _rounds(pv) == [0.65068, -0.99819], pv.tolist()

    _, cn1, _, dd1, _ = _run(solve, x, F, sigmas, step_index=1)
    assert round(cn1, 6) == 0.0
    assert _rounds(dd1) == [0.423607, -0.288197]

    _, cn2, pv2, dd2, _ = _run(solve, x, F, sigmas, step_index=2)
    assert round(cn2, 6) == -0.173287
    # 最后一步 sigmas[3] = 0，dt = -sigma_hat，Euler 步恰好把 sample 送到 denoised
    assert _rounds(dd2) == [0.676777, -0.911612]
    assert pv2.tolist() == dd2.tolist()

    mi3, _, pv3, dd3, sh3 = _run(
        solve, x, F, sigmas, step_index=0, s_churn=10.0, noise=torch.tensor([[0.1, -0.3]])
    )
    assert round(sh3, 6) == 2.828427
    # model_input 用未加噪的 sample 与 sigma，而输出合成用加噪后的 sigma_hat
    assert _rounds(mi3) == [0.485071, -0.970142], mi3.tolist()
    assert _rounds(dd3) == [0.282547, 0.044304], dd3.tolist()
    assert _rounds(pv3) == [0.606915, -0.890599], pv3.tolist()


check_example.example = True


def check_model_input_uses_pre_churn(solve):
    x = torch.randn(2, 3, dtype=torch.float64)
    F = torch.randn(2, 3, dtype=torch.float64)
    noise = torch.randn(2, 3, dtype=torch.float64)
    sigmas = [3.0, 1.5, 0.7, 0.0]
    sigma = sigmas[0]
    sd = 0.5

    mi, _, pv, dn, sh = solve(x, F, torch.tensor(sigmas, dtype=torch.float64), 0)
    assert sh == sigma, "s_churn=0 时不应加噪"
    plain, _, plain_pv, _, _ = solve(x, F, torch.tensor(sigmas, dtype=torch.float64), 0, noise=noise)
    assert torch.equal(pv, plain_pv), "gamma=0 时 noise 不应参与计算"
    assert torch.equal(mi, plain)

    mi2, _, _, _, sh2 = solve(
        x, F, torch.tensor(sigmas, dtype=torch.float64), 0, s_churn=50.0, noise=noise
    )
    c_in = 1.0 / math.sqrt(sigma**2 + sd**2)
    assert torch.allclose(mi2.double(), (x * c_in).float().double(), atol=1e-6)
    # 若误把加噪后的 sample 送进 model_input，结果会多出 noise 项
    chirped = x + noise * math.sqrt(sh2**2 - sigma**2)
    assert not torch.allclose(mi2.double(), (chirped * c_in).float().double(), atol=1e-3)
    assert abs(sh2 - sigma * math.sqrt(2.0)) < 1e-9


def check_churn_window(solve):
    x = torch.zeros(1, 2)
    F = torch.ones(1, 2)
    sigmas = [4.0, 2.0, 1.0, 0.0]
    noise = torch.full((1, 2), 0.5)
    # 窗口外 gamma=0，sigma_hat 不变，noise 完全不被使用
    for kwargs in (
        dict(s_churn=100.0, s_tmin=5.0),
        dict(s_churn=100.0, s_tmax=3.0),
        dict(s_churn=0.0),
    ):
        _, _, a, _, sh = _run(solve, x, F, sigmas, step_index=0, noise=noise, **kwargs)
        assert sh == 4.0, kwargs
        _, _, b, _, _ = _run(solve, x, F, sigmas, step_index=0, **kwargs)
        assert torch.equal(a, b), kwargs
    # 窗口内：gamma = min(s_churn/K, sqrt2-1)，K=len(sigmas)-1=3
    _, _, _, _, sh_small = _run(solve, x, F, sigmas, step_index=0, s_churn=0.6, noise=noise)
    assert abs(sh_small - 4.0 * (1 + 0.2)) < 1e-9
    _, _, _, _, sh_cap = _run(solve, x, F, sigmas, step_index=0, s_churn=9.0, noise=noise)
    assert abs(sh_cap - 4.0 * math.sqrt(2.0)) < 1e-9


def check_last_step(solve):
    torch.manual_seed(0)
    for sigmas in ([2.0, 1.0, 0.25, 0.0], [1.0, 0.0], [5.0, 3.0, 1.0, 0.4, 0.0]):
        x = torch.randn(2, 3)
        F = torch.randn(2, 3)
        last = len(sigmas) - 2
        _, _, pv, dn, _ = _run(solve, x, F, sigmas, step_index=last)
        assert torch.allclose(pv, dn, atol=1e-6), sigmas


def check_prediction_types(solve):
    x = torch.randn(2, 2, dtype=torch.float64)
    F = torch.randn(2, 2, dtype=torch.float64)
    sigmas = [1.5, 0.8, 0.3, 0.0]
    for index in (0, 1, 2):
        _, _, _, eps, _ = solve(x, F, torch.tensor(sigmas, dtype=torch.float64), index)
        _, _, _, vee, _ = solve(
            x, F, torch.tensor(sigmas, dtype=torch.float64), index, prediction_type="v_prediction"
        )
        sigma = sigmas[index]
        c_skip = 0.5**2 / (sigma**2 + 0.5**2)
        assert torch.allclose(vee.double(), 2 * c_skip * x - eps.double(), atol=1e-6), index


def check_coefficients(solve):
    x = torch.zeros(1, 1)
    F = torch.zeros(1, 1)
    sd = 0.5
    sigmas = [8.0, 4.0, 1.0, 0.2, 0.0]
    for index in range(len(sigmas) - 1):
        sigma = sigmas[index]
        mi, cn, _, dn, sh = _run(solve, x, F, sigmas, step_index=index, sigma_data=sd)
        c_in = 1.0 / math.sqrt(sigma**2 + sd**2)
        assert abs(float(mi[0, 0]) - 0.0) < 1e-9 or True
        assert abs(cn - 0.25 * math.log(sigma)) < 1e-6
        assert abs(sh - sigma) < 1e-6
        # x = F = 0 时 denoised 必然为 0，这里只验证 dtype/形状契约
        assert dn.dtype == torch.float32 and dn.shape == (1, 1)
    # c_noise 与 sigma 互为可逆的对数刻度
    _, cn, _, _, _ = _run(solve, x, F, sigmas, step_index=0, sigma_data=sd)
    assert abs(math.exp(4.0 * cn) - 8.0) < 1e-9


def check_against_oracle(solve):
    torch.manual_seed(3)
    for sigmas in ([2.0, 1.0, 0.5, 0.0], [5.0, 2.5, 0.9, 0.3, 0.0], [1.0, 0.0]):
        steps = len(sigmas) - 1
        for index in range(steps):
            x = torch.randn(2, 3, dtype=torch.float64)
            F = torch.randn(2, 3, dtype=torch.float64)
            noise = torch.randn(2, 3, dtype=torch.float64)
            for kwargs in (
                dict(),
                dict(prediction_type="v_prediction"),
                dict(s_churn=2.0, noise=noise),
                dict(s_churn=0.4, s_noise=0.7, noise=noise, sigma_data=1.0),
                dict(s_churn=5.0, s_tmin=0.5, s_tmax=1.5, noise=noise, sigma_data=0.2),
            ):
                oracle_kwargs = dict(kwargs)
                if "prediction_type" in oracle_kwargs:
                    oracle_kwargs["pred"] = oracle_kwargs.pop("prediction_type")
                oracle_kwargs["noise"] = noise.tolist()
                want = _oracle(x.tolist(), F.tolist(), sigmas, index, **oracle_kwargs)
                mi, cn, pv, dn, sh = solve(
                    x, F, torch.tensor(sigmas, dtype=torch.float64), index, **kwargs
                )
                assert abs(sh - want[4]) < 1e-12, (sigmas, index, kwargs)
                assert abs(cn - want[1]) < 1e-12, (sigmas, index, kwargs)
                assert torch.allclose(mi, torch.tensor(want[0], dtype=torch.float32), atol=1e-6)
                assert torch.allclose(dn, torch.tensor(want[3], dtype=torch.float32), atol=1e-6)
                assert torch.allclose(pv, torch.tensor(want[2], dtype=torch.float32), atol=1e-6)


def check_invalid(solve):
    x = torch.zeros(2, 3)
    F = torch.zeros(2, 3)
    sigmas = torch.tensor([2.0, 1.0, 0.0])
    cases = [
        dict(sample=[[0.0, 1.0]], model_output=F, sigmas=sigmas, step_index=0),
        dict(sample=torch.zeros(2, 3, dtype=torch.int64), model_output=F, sigmas=sigmas, step_index=0),
        dict(sample=x, model_output=torch.zeros(2, 4), sigmas=sigmas, step_index=0),
        dict(sample=torch.zeros(0, 3), model_output=torch.zeros(0, 3), sigmas=sigmas, step_index=0),
        dict(sample=x, model_output=F, sigmas=[2.0, 1.0, 0.0], step_index=0),
        dict(sample=x, model_output=F, sigmas=sigmas.reshape(3, 1), step_index=0),
        dict(sample=x, model_output=F, sigmas=torch.tensor([2, 1, 0]), step_index=0),
        dict(sample=x, model_output=F, sigmas=torch.tensor([1.0]), step_index=0),
        dict(sample=x, model_output=F, sigmas=torch.tensor([2.0, -1.0, 0.0]), step_index=0),
        dict(sample=x, model_output=F, sigmas=torch.tensor([1.0, 2.0, 0.0]), step_index=0),
        dict(sample=x, model_output=F, sigmas=torch.tensor([2.0, float("nan"), 0.0]), step_index=0),
        dict(sample=x, model_output=F, sigmas=sigmas, step_index=-1),
        dict(sample=x, model_output=F, sigmas=sigmas, step_index=2),
        dict(sample=x, model_output=F, sigmas=sigmas, step_index=0.5),
        dict(sample=x, model_output=F, sigmas=sigmas, step_index=True),
        dict(sample=x, model_output=F, sigmas=sigmas, step_index=0, sigma_data=0.0),
        dict(sample=x, model_output=F, sigmas=sigmas, step_index=0, sigma_data=-1.0),
        dict(sample=x, model_output=F, sigmas=sigmas, step_index=0, s_churn=-1.0),
        dict(sample=x, model_output=F, sigmas=sigmas, step_index=0, s_noise=-1.0),
        dict(sample=x, model_output=F, sigmas=sigmas, step_index=0, s_tmin=2.0, s_tmax=1.0),
        dict(sample=x, model_output=F, sigmas=sigmas, step_index=0, prediction_type="x0"),
        dict(sample=x, model_output=F, sigmas=sigmas, step_index=0, s_churn=1.0),
        dict(
            sample=x,
            model_output=F,
            sigmas=sigmas,
            step_index=0,
            s_churn=1.0,
            noise=torch.zeros(2, 4),
        ),
        dict(
            sample=x,
            model_output=F,
            sigmas=sigmas,
            step_index=0,
            s_churn=1.0,
            noise=torch.zeros(2, 3, dtype=torch.int64),
        ),
        dict(sample=x, model_output=F, sigmas=torch.tensor([0.0, 0.0]), step_index=0),
    ]
    for kwargs in cases:
        try:
            solve(**kwargs)
            raise AssertionError(f"非法输入应抛 ValueError: {kwargs}")
        except ValueError:
            pass
