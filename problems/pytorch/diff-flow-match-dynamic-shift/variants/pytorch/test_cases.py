import math

import torch


def _oracle(
    steps,
    shift=1.0,
    dynamic=False,
    mu=None,
    kind="exponential",
    shift_terminal=None,
    invert=False,
    sigma_min=0.0,
    sigma_max=1.0,
    train=1000,
):
    """独立 oracle：python 标量逐项重做 linspace / 平移 / 拉伸 / 反相 / 终端。"""
    if steps == 1:
        sigmas = [sigma_max]
    else:
        sigmas = [sigma_max + (sigma_min - sigma_max) * i / (steps - 1) for i in range(steps)]

    if dynamic:
        if kind == "exponential":
            scale = math.exp(mu)
            shifted = []
            for x in sigmas:
                shifted.append(0.0 if x == 0.0 else scale / (scale + (1.0 / x - 1.0)))
        else:
            shifted = []
            for x in sigmas:
                shifted.append(0.0 if x == 0.0 else mu / (mu + (1.0 / x - 1.0)))
        sigmas = shifted
    else:
        sigmas = [shift * x / (1.0 + (shift - 1.0) * x) for x in sigmas]

    if shift_terminal is not None:
        scale = (1.0 - sigmas[-1]) / (1.0 - shift_terminal)
        sigmas = [1.0 - (1.0 - x) / scale for x in sigmas]

    timesteps = [x * train for x in sigmas]
    if invert:
        sigmas = [1.0 - x for x in sigmas]
        timesteps = [x * train for x in sigmas]
        sigmas = sigmas + [1.0]
    else:
        sigmas = sigmas + [0.0]
    return timesteps, sigmas


def _run(
    solve,
    steps,
    shift=1.0,
    dynamic=False,
    mu=None,
    kind="exponential",
    shift_terminal=None,
    invert=False,
    sigma_min=0.0,
    sigma_max=1.0,
    train=1000,
):
    return solve(
        steps,
        shift=shift,
        use_dynamic_shifting=dynamic,
        mu=mu,
        time_shift_type=kind,
        shift_terminal=shift_terminal,
        invert_sigmas=invert,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        num_train_timesteps=train,
    )


def _rounds(values):
    return [round(float(v), 6) for v in values]


def check_example(solve):
    t, s = _run(solve, 4, shift=2.0)
    assert t.dtype == torch.float32 and s.dtype == torch.float32
    assert t.shape == (4,) and s.shape == (5,)
    assert _rounds(t) == [1000.0, 800.0, 500.0, 0.0], t.tolist()
    # 末项 σ = 0 与追加的终端 0 是两个不同的位置
    assert _rounds(s) == [1.0, 0.8, 0.5, 0.0, 0.0], s.tolist()

    t, s = _run(solve, 4, dynamic=True, mu=1.0)
    assert _rounds(t) == [1000.0, 844.637573, 576.116882, 0.0], t.tolist()
    assert _rounds(s) == [1.0, 0.844638, 0.576117, 0.0, 0.0], s.tolist()

    t, s = _run(solve, 4, shift=2.0, shift_terminal=0.2)
    assert _rounds(t) == [1000.0, 840.0, 600.0, 200.0], t.tolist()
    assert _rounds(s) == [1.0, 0.84, 0.6, 0.2, 0.0], s.tolist()

    t, s = _run(solve, 4, shift=2.0, invert=True)
    assert _rounds(t) == [0.0, 200.0, 500.0, 1000.0], t.tolist()
    assert _rounds(s) == [0.0, 0.2, 0.5, 1.0, 1.0], s.tolist()

    t, s = _run(solve, 4, shift=2.0, shift_terminal=0.2, invert=True)
    assert _rounds(t) == [0.0, 160.0, 400.0, 800.0], t.tolist()
    assert _rounds(s) == [0.0, 0.16, 0.4, 0.8, 1.0], s.tolist()


check_example.example = True


def check_static_dynamic_equivalence(solve):
    for steps in (2, 4, 7):
        for shift in (1.0, 1.5, 2.0, 5.0):
            static, _ = _run(solve, steps, shift=shift)
            linear, _ = _run(solve, steps, dynamic=True, mu=shift, kind="linear")
            expo, _ = _run(solve, steps, dynamic=True, mu=math.log(shift))
            assert torch.allclose(static, linear, atol=1e-5), (steps, shift)
            assert torch.allclose(static, expo, atol=1e-5), (steps, shift)
    # 恒等：静态 shift=1 与指数平移 mu=0
    sigmas0, _ = _run(solve, 5, shift=1.0)
    for kw in (dict(shift=1.0), dict(dynamic=True, mu=0.0)):
        t, _ = _run(solve, 5, **kw)
        assert torch.allclose(t, sigmas0, atol=1e-6), kw


def check_monotone_and_endpoints(solve):
    for steps in (2, 5, 9):
        for kw in (
            dict(shift=3.0),
            dict(dynamic=True, mu=1.7),
            dict(dynamic=True, mu=0.8, kind="linear"),
            dict(shift=3.0, shift_terminal=0.6),
        ):
            t, s = _run(solve, steps, **kw)
            core = s[:-1]
            assert bool((core[1:] <= core[:-1] + 1e-7).all()), kw
            assert bool((core >= -1e-7).all()) and bool((core <= 1.0 + 1e-7).all()), kw
            assert torch.allclose(t, core * 1000.0, atol=1e-3), kw
            # 无 terminal 时两端点被平移固定
            if "shift_terminal" not in kw:
                assert abs(float(core[0]) - 1.0) < 1e-6, kw
                assert abs(float(core[-1]) - 0.0) < 1e-6, kw


def check_terminal_ordering(solve):
    steps = 5
    _, plain = _run(solve, steps, shift=2.0, sigma_max=0.9)
    shifted = plain[:-1]
    t, s = _run(solve, steps, shift=2.0, shift_terminal=0.25, sigma_max=0.9)
    core = s[:-1]
    # 拉伸只把末项钉到 terminal，形状是 1 - (1-z)/scale
    assert abs(float(core[-1]) - 0.25) < 1e-6
    scale = (1.0 - float(shifted[-1])) / 0.75
    want = 1.0 - (1.0 - shifted.double()) / scale
    assert torch.allclose(core.double(), want, atol=1e-6), (core.tolist(), want.tolist())
    # timesteps 用的是拉伸之后的 σ
    assert abs(float(t[-1]) - 250.0) < 1e-3
    # 顺序反了（先拉伸再平移）会得到不同的末项，暴露这处写法
    pre = 1.0 - (1.0 - shifted.double()) / scale
    pre = 2.0 * pre / (1.0 + pre)
    assert abs(float(pre[-1]) - 0.25) > 1e-3
    # 单调仍在，且首项被拉伸推到更靠噪声侧
    assert bool((core[1:] <= core[:-1] + 1e-7).all())
    assert float(core[0]) > float(shifted[0]) + 1e-4


def check_invert_sigmas(solve):
    for steps in (2, 4, 6):
        for kw in (dict(shift=2.0), dict(shift=2.0, shift_terminal=0.3), dict(dynamic=True, mu=1.2)):
            _, plain_sig = _run(solve, steps, **kw)
            t, s = _run(solve, steps, invert=True, **kw)
            assert torch.allclose(s[:-1], 1.0 - plain_sig[:-1], atol=1e-6), kw
            assert float(s[-1]) == 1.0
            assert torch.allclose(t, s[:-1] * 1000.0, atol=1e-3), kw
            assert bool((s[1:-1] >= s[:-2] + 1e-7).all()), kw


def check_sigma_range(solve):
    t, s = _run(solve, 4, shift=2.0, sigma_min=0.1, sigma_max=0.9)
    assert _rounds(s) == [0.947368, 0.77551, 0.536585, 0.181818, 0.0], s.tolist()
    assert _rounds(t) == [947.368408, 775.510193, 536.585388, 181.818176, 0.0][:4], t.tolist()

    one, s1 = _run(solve, 1, shift=2.0)
    assert one.tolist() == [1000.0]
    assert s1.tolist() == [1.0, 0.0]


def check_train_scaling(solve):
    for train in (1, 10, 1000, 4096):
        t, s = _run(solve, 4, shift=1.5, train=train)
        assert torch.allclose(t, s[:-1] * float(train), atol=1e-2), train


def check_against_oracle(solve):
    for steps in (1, 2, 3, 5, 8):
        for kw in (
            dict(shift=1.0),
            dict(shift=2.5),
            dict(dynamic=True, mu=0.5),
            dict(dynamic=True, mu=3.0, kind="linear"),
            dict(shift=1.8, shift_terminal=0.4),
            dict(shift=1.8, invert=True),
            dict(dynamic=True, mu=-0.7),
            dict(shift=4.0, sigma_min=0.05, sigma_max=0.8, shift_terminal=0.2, invert=True),
        ):
            if steps == 1 and "shift_terminal" in kw:
                continue
            want_t, want_s = _oracle(steps, **kw)
            t, s = _run(solve, steps, **kw)
            assert torch.allclose(t, torch.tensor(want_t, dtype=torch.float32), atol=1e-4), (steps, kw)
            assert torch.allclose(s, torch.tensor(want_s, dtype=torch.float32), atol=1e-6), (steps, kw)


def check_invalid(solve):
    cases = [
        dict(num_inference_steps=0),
        dict(num_inference_steps=2.0),
        dict(num_inference_steps=True),
        dict(num_inference_steps=4, shift=0.0),
        dict(num_inference_steps=4, shift=-1.0),
        dict(num_inference_steps=4, shift=float("nan")),
        dict(num_inference_steps=4, time_shift_type="cosine"),
        dict(num_inference_steps=4, use_dynamic_shifting=True),
        dict(num_inference_steps=4, use_dynamic_shifting=True, mu=float("inf")),
        dict(num_inference_steps=4, use_dynamic_shifting=True, mu=0.0, time_shift_type="linear"),
        dict(num_inference_steps=4, shift_terminal=0.0),
        dict(num_inference_steps=4, shift_terminal=1.0),
        dict(num_inference_steps=4, shift_terminal=-0.2),
        dict(num_inference_steps=1, shift_terminal=0.5),
        dict(num_inference_steps=4, sigma_min=0.5, sigma_max=0.5),
        dict(num_inference_steps=4, sigma_min=-0.1),
        dict(num_inference_steps=4, sigma_max=1.5),
        dict(num_inference_steps=4, num_train_timesteps=0),
    ]
    for kwargs in cases:
        try:
            solve(**kwargs)
            raise AssertionError(f"非法输入应抛 ValueError: {kwargs}")
        except ValueError:
            pass
