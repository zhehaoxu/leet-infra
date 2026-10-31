import math

import numpy as np
import torch


def _oracle_solve(
    sample,
    model_output,
    sigmas,
    step_index,
    history,
    solver_order=2,
    lower_order_final=True,
    prediction_type="epsilon",
    use_flow_sigmas=False,
):
    """独立 numpy 重写（不 import solution），公式从 2204.13902 与 diffusers
    scheduling_deis_multistep.py 的数学定义直接推出。"""
    x = np.asarray(sample, dtype=np.float64)
    mo = np.asarray(model_output, dtype=np.float64)
    s = np.asarray(sigmas, dtype=np.float64)
    n = len(s) - 1

    def alpha_sigma(v):
        if use_flow_sigmas:
            return 1.0 - v, v
        a = 1.0 / np.sqrt(v * v + 1.0)
        return a, v * a

    sc = s[step_index]
    ac, sic = alpha_sigma(sc)
    if prediction_type == "epsilon":
        x0 = x - sic * mo
    elif prediction_type == "sample":
        x0 = mo
    elif prediction_type == "v_prediction":
        x0 = ac * x - sic * mo
    else:
        x0 = x - sc * mo
    d = (x - ac * x0) / sic

    lof = (step_index == n - 1) and lower_order_final and n < 15
    los = (step_index == n - 2) and lower_order_final and n < 15
    low = len(history)
    if solver_order == 1 or low < 1 or lof:
        used = 1
    elif solver_order == 2 or low < 2 or los:
        used = 2
    else:
        used = 3

    if used == 1:
        at, st = alpha_sigma(s[step_index + 1])
        asg, ss = alpha_sigma(s[step_index])
        h = (np.log(at) - np.log(st)) - (np.log(asg) - np.log(ss))
        xt = (at / asg) * x - st * (np.exp(h) - 1.0) * d
        coefs = []
    else:
        at, st = alpha_sigma(s[step_index + 1])
        a0, s0 = alpha_sigma(s[step_index])
        a1, s1 = alpha_sigma(s[step_index - 1])
        rt, r0, r1 = st / at, s0 / a0, s1 / a1
        if used == 2:
            def ind(t, b, c):
                return t * (-np.log(c) + np.log(t) - 1.0) / (np.log(b) - np.log(c))

            c1 = ind(rt, r0, r1) - ind(r0, r0, r1)
            c2 = ind(rt, r1, r0) - ind(r0, r1, r0)
            xt = at * (x / a0 + c1 * d + c2 * np.asarray(history[-1]))
            coefs = [c1, c2]
        else:
            a2, s2 = alpha_sigma(s[step_index - 2])
            r2 = s2 / a2

            def ind(t, b, c, dd):
                num = t * (
                    np.log(c) * (np.log(dd) - np.log(t) + 1.0)
                    - np.log(dd) * np.log(t)
                    + np.log(dd)
                    + np.log(t) ** 2
                    - 2.0 * np.log(t)
                    + 2.0
                )
                return num / ((np.log(b) - np.log(c)) * (np.log(b) - np.log(dd)))

            c1 = ind(rt, r0, r1, r2) - ind(r0, r0, r1, r2)
            c2 = ind(rt, r1, r2, r0) - ind(r0, r1, r2, r0)
            c3 = ind(rt, r2, r0, r1) - ind(r0, r2, r0, r1)
            xt = at * (
                x / a0 + c1 * d + c2 * np.asarray(history[-1]) + c3 * np.asarray(history[-2])
            )
            coefs = [c1, c2, c3]

    return d, used, xt, coefs


def check_first_order_hand(solve):
    """手算硬编码：N=2, sigmas=[8,2,0.5]，step 0 一阶。

    σ_s=8: α_s=1/√65, σ_s2=8/√65 → λ_s = −ln 8
    σ_t=2: α_t=1/√5,  σ_t2=2/√5  → λ_t = −ln 2
    h = ln 4, e^h−1 = 3
    convert: x0 = sample − (8/√65)·mo；d = (sample − α_s·x0)/(8/√65)
    x = √13·sample − (2/√5)·3·d
    """
    sample = torch.tensor([2.0, -1.0])
    mo = torch.tensor([1.0, 1.0])
    sigmas = torch.tensor([8.0, 2.0, 0.5])
    got = solve(sample, mo, sigmas, 0, [])
    assert got["used_order"] == 1 and got["coefs"] == []

    a_s = 1.0 / math.sqrt(65.0)
    sig_s = 8.0 / math.sqrt(65.0)
    x0 = sample.to(torch.float64) - sig_s * mo.to(torch.float64)
    d = (sample.to(torch.float64) - a_s * x0) / sig_s
    want = math.sqrt(13.0) * sample.to(torch.float64) - (2.0 / math.sqrt(5.0)) * 3.0 * d
    assert torch.allclose(got["x_next"].to(torch.float64), want, atol=1e-6), got["x_next"]
    # 手算数值锚点
    assert torch.allclose(
        got["x_next"], torch.tensor([2.140776, -1.569618]), atol=1e-5
    ), got["x_next"]
    # converted 锚点：d = [1.889599, -0.758747]
    assert torch.allclose(
        got["converted"], torch.tensor([1.889599, -0.758747], dtype=torch.float64), atol=1e-5
    )


def check_matches_oracle_random(solve):
    """三阶全配置随机对拍。"""
    g = torch.Generator().manual_seed(20261031)
    for trial in range(8):
        n = int(torch.randint(5, 8, (1,), generator=g))
        base = torch.rand(n + 1, generator=g) * 8.0 + 0.1
        sigmas = torch.sort(base, descending=True).values
        sample = torch.randn(3, generator=g)
        mo = torch.randn(3, generator=g)
        order = 3
        step = n - 3  # 非末两步且 history 满 → 走三阶
        history = [torch.randn(3, generator=g) for _ in range(min(step, 3))]
        d, used, xt, coefs = _oracle_solve(
            sample.numpy(), mo.numpy(), sigmas.numpy(), step, [h.numpy() for h in history],
            solver_order=order,
        )
        got = solve(sample, mo, sigmas, step, history, solver_order=order)
        assert got["used_order"] == 3
        assert torch.allclose(got["converted"], torch.tensor(d), atol=1e-9)
        assert torch.allclose(got["x_next"].to(torch.float64), torch.tensor(xt), atol=1e-9)
        assert len(got["coefs"]) == 3
        for a, b in zip(got["coefs"], coefs):
            assert abs(a - b) < 1e-9


def check_second_order_and_ramp(solve):
    """二阶系数与预热/末两步降阶账（与 520 的 order 表独立：这里看数值）。"""
    sigmas = torch.tensor([10.0, 4.0, 1.0, 0.5])  # N=3
    sample = torch.tensor([1.0, 2.0])
    mo = torch.tensor([0.5, -0.5])
    h1 = torch.tensor([0.3, -0.2])
    h2 = torch.tensor([0.1, 0.1])

    # step 1, history 1 条 → 二阶
    d, used, xt, coefs = _oracle_solve(
        sample.numpy(), mo.numpy(), sigmas.numpy(), 1, [h1.numpy()]
    )
    got = solve(sample, mo, sigmas, 1, [h1], solver_order=3)
    assert got["used_order"] == 2
    assert torch.allclose(got["x_next"].to(torch.float64), torch.tensor(xt), atol=1e-9)
    assert abs(got["coefs"][0] - coefs[0]) < 1e-12 and abs(got["coefs"][1] - coefs[1]) < 1e-12

    # 末步（step 2, N=3<15）→ 强制一阶，即便 history 已满
    got_last = solve(sample, mo, sigmas, 2, [h1, h2], solver_order=3)
    assert got_last["used_order"] == 1

    # 末步（step 2, N=3）在 lower_order_final=False 且 history 满 2 条时走三阶
    got_nolof = solve(sample, mo, sigmas, 2, [h1, h2], solver_order=3, lower_order_final=False)
    assert got_nolof["used_order"] == 3

    # N=15 时末步不降阶：step 14 走三阶（history 满 3）
    big = torch.linspace(12.0, 0.6, 16)  # N=15
    h3 = torch.tensor([0.0, 0.0])
    got15 = solve(sample, mo, big, 14, [h1, h2, h3], solver_order=3)
    assert got15["used_order"] == 3
    # N=14 时末步（step 13）强制一阶
    small = big[:-1].clone()
    got14 = solve(sample, mo, small, 13, [h1, h2, h3], solver_order=3)
    assert got14["used_order"] == 1


def check_prediction_types(solve):
    """sample / v_prediction / flow_prediction 转换分支 + flow 的 α/σ 变换。"""
    sample = torch.tensor([2.0, -1.0])
    mo = torch.tensor([1.0, 0.5])
    sigmas = torch.tensor([4.0, 1.0, 0.25])

    d_s, _, xt_s, _ = _oracle_solve(
        sample.numpy(), mo.numpy(), sigmas.numpy(), 0, [], prediction_type="sample"
    )
    got = solve(sample, mo, sigmas, 0, [], prediction_type="sample")
    assert torch.allclose(got["converted"], torch.tensor(d_s), atol=1e-12)
    assert torch.allclose(got["x_next"].to(torch.float64), torch.tensor(xt_s), atol=1e-9)

    d_v, _, xt_v, _ = _oracle_solve(
        sample.numpy(), mo.numpy(), sigmas.numpy(), 0, [], prediction_type="v_prediction"
    )
    got_v = solve(sample, mo, sigmas, 0, [], prediction_type="v_prediction")
    assert torch.allclose(got_v["converted"], torch.tensor(d_v), atol=1e-12)

    # flow 口径要求 sigma ∈ (0,1)（上游 flow sigmas 由 1-α 生成）
    flow_sigmas = torch.tensor([0.8, 0.3, 0.05])
    d_f, _, xt_f, _ = _oracle_solve(
        sample.numpy(), mo.numpy(), flow_sigmas.numpy(), 0, [], use_flow_sigmas=True,
        prediction_type="flow_prediction",
    )
    got_f = solve(sample, mo, flow_sigmas, 0, [], use_flow_sigmas=True, prediction_type="flow_prediction")
    assert torch.allclose(got_f["converted"], torch.tensor(d_f), atol=1e-12)
    assert torch.allclose(got_f["x_next"].to(torch.float64), torch.tensor(xt_f), atol=1e-9)


def check_validation(solve):
    sample = torch.tensor([1.0, 2.0])
    mo = torch.tensor([0.5, -0.5])
    sigmas = torch.tensor([10.0, 4.0, 1.0, 0.5])
    h1 = torch.tensor([0.3, -0.2])

    bad = [
        (lambda: solve(sample, mo, torch.tensor([1.0]), 0, []), "sigmas 太短"),
        (lambda: solve(sample, mo, torch.tensor([4.0, 1.0, 0.0]), 0, []), "含 0"),
        (lambda: solve(sample, mo, torch.tensor([1.0, 4.0, 0.5]), 0, []), "单调递增"),
        (lambda: solve(sample, mo, sigmas, 3, []), "step_index 越界"),
        (lambda: solve(sample, mo, sigmas, 0, [h1]), "history 应为 0 条"),
        (lambda: solve(sample, mo, sigmas, 1, []), "history 应为 1 条"),
        (lambda: solve(sample, mo, sigmas, 1, [torch.randn(3)]), "history 形状不符"),
        (lambda: solve(sample, mo, sigmas, 0, [], solver_order=4), "非法阶数"),
        (lambda: solve(sample, mo, sigmas, 0, [], prediction_type="flow"), "未知类型"),
        (lambda: solve(sample, torch.randn(2, 2), sigmas, 0, []), "形状不一致"),
    ]
    for fn, tag in bad:
        try:
            fn()
        except ValueError:
            pass
        else:
            raise AssertionError(f"用例未抛错：{tag}")
