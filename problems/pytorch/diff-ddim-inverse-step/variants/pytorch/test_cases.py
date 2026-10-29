import math

import torch


def _oracle_one(ac, num_train, num_inf, timestep, x, m, ptype, clip, crange, alpha_one):
    prev_t = timestep
    t = min(timestep - num_train // num_inf, num_train - 1)
    a_t = ac[t] if t >= 0 else (1.0 if alpha_one else ac[0])
    a_prev = ac[prev_t]
    b_t = 1.0 - a_t
    if ptype == "epsilon":
        x0 = (x - math.sqrt(b_t) * m) / math.sqrt(a_t)
        eps = m
    elif ptype == "sample":
        x0 = m
        eps = (x - math.sqrt(a_t) * x0) / math.sqrt(b_t)
    else:
        x0 = math.sqrt(a_t) * x - math.sqrt(b_t) * m
        eps = math.sqrt(a_t) * m + math.sqrt(b_t) * x
    if clip:
        x0 = max(-crange, min(crange, x0))
    prev = math.sqrt(a_prev) * x0 + math.sqrt(1.0 - a_prev) * eps
    return prev, x0


def _check(solve, ac_list, timestep, sample, model_output, num_inf, **kw):
    ac = torch.tensor(ac_list, dtype=torch.float64)
    prev, x0 = solve(ac, timestep, sample, model_output, num_inf, **kw)
    assert prev.dtype == torch.float32 and x0.dtype == torch.float32
    assert prev.shape == sample.shape and x0.shape == sample.shape
    xs = sample.tolist()
    ms = model_output.tolist()

    def flat(v):
        if isinstance(v, list):
            for item in v:
                yield from flat(item)
        else:
            yield v

    ptype = kw.get("prediction_type", "epsilon")
    clip = kw.get("clip_sample", True)
    crange = kw.get("clip_sample_range", 1.0)
    alpha_one = kw.get("set_alpha_to_one", True)
    want = [
        _oracle_one(ac_list, len(ac_list), num_inf, timestep, x, m, ptype, clip, crange, alpha_one)
        for x, m in zip(flat(xs), flat(ms))
    ]
    got_prev = list(flat(prev.tolist()))
    got_x0 = list(flat(x0.tolist()))
    for i, ((wp, wx), gp, gx) in enumerate(zip(want, got_prev, got_x0)):
        assert abs(gp - wp) <= 1e-5 * max(1.0, abs(wp)), (i, gp, wp)
        assert abs(gx - wx) <= 1e-5 * max(1.0, abs(wx)), (i, gx, wx)
    return prev, x0


def check_example(solve):
    ac = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    sample = torch.tensor(2.0)
    model_output = torch.tensor(0.5)

    prev, x0 = solve(ac, 5, sample, model_output, 10, clip_sample=False)
    assert [round(float(v), 6) for v in (prev, x0)] == [1.859925, 2.328427], (prev, x0)

    prev_c, x0_c = solve(ac, 5, sample, model_output, 10)
    assert [round(float(v), 6) for v in (prev_c, x0_c)] == [1.019754, 1.0], (prev_c, x0_c)


check_example.example = True


def check_initial_alpha(solve):
    ac = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
    sample = torch.tensor(1.0)
    mo = torch.tensor(0.2)

    prev_one, x0_one = _check(solve, ac, 0, sample, mo, 10)
    assert [round(float(v), 6) for v in (prev_one, x0_one)] == [1.011929, 1.0], (prev_one, x0_one)

    prev_first, x0_first = _check(solve, ac, 0, sample, mo, 10, set_alpha_to_one=False)
    assert [round(float(v), 6) for v in (prev_first, x0_first)] == [1.0, 0.987426], (prev_first, x0_first)

    # 步长比例大于 1 时，索引 0 之下仍有更细的层级；两种初始 alpha 必须给出不同结果
    prev_mid, _ = _check(solve, ac, 2, sample, mo, 5)
    assert abs(float(prev_mid) - float(prev_one)) > 1e-3, "不同 timestep 不应得到同一结果"


def check_noise_level_roundtrip(solve):
    ac = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
    x_true, eps_true = 0.7, -0.3
    num_inf = 10
    timestep = 5
    t = timestep - len(ac) // num_inf
    x_t = math.sqrt(ac[t]) * x_true + math.sqrt(1.0 - ac[t]) * eps_true

    prev, x0 = _check(
        solve, ac, timestep, torch.tensor(x_t, dtype=torch.float32), torch.tensor(eps_true, dtype=torch.float32),
        num_inf, clip_sample=False,
    )
    want = math.sqrt(ac[timestep]) * x_true + math.sqrt(1.0 - ac[timestep]) * eps_true
    assert abs(float(prev) - want) < 1e-5, (prev, want)
    assert abs(float(x0) - x_true) < 1e-5, x0

    v_true = math.sqrt(ac[t]) * eps_true - math.sqrt(1.0 - ac[t]) * x_true
    prev_v, x0_v = _check(
        solve, ac, timestep, torch.tensor(x_t, dtype=torch.float32), torch.tensor(v_true, dtype=torch.float32),
        num_inf, prediction_type="v_prediction", clip_sample=False,
    )
    assert abs(float(prev_v) - want) < 1e-5, (prev_v, want)
    assert abs(float(x0_v) - x_true) < 1e-5, x0_v


def check_sample_prediction(solve):
    ac = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
    prev, x0 = _check(solve, ac, 5, torch.tensor(0.3), torch.tensor(0.8), 10, prediction_type="sample", clip_sample=False)
    assert [round(float(v), 6) for v in (prev, x0)] == [0.214921, 0.8], (prev, x0)

    try:
        solve(torch.tensor(ac), 0, torch.tensor(1.0), torch.tensor(0.5), 10, prediction_type="sample")
    except ValueError:
        pass
    else:
        raise AssertionError("sample 预测类型遇到 beta_prod_t=0 应抛 ValueError")


def check_clip(solve):
    ac = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
    sample = torch.tensor([3.0, -2.0, 0.4])
    mo = torch.tensor([0.0, 0.0, 0.0])

    raw_prev, raw_x0 = _check(solve, ac, 6, sample, mo, 10, clip_sample=False)
    assert bool((raw_x0.abs() > 1.0).any()), "该用例本应产生越界的 x0"

    prev, x0 = _check(solve, ac, 6, sample, mo, 10, clip_sample=True)
    assert float(x0.max()) <= 1.0 + 1e-6 and float(x0.min()) >= -1.0 - 1e-6, x0
    assert abs(float(x0[0]) - 1.0) < 1e-6 and abs(float(x0[1]) + 1.0) < 1e-6, x0
    assert abs(float(x0[2]) - float(raw_x0[2])) < 1e-6, "未越界的位置不应被截断"
    assert not torch.allclose(prev, raw_prev)

    tight, tight_x0 = _check(solve, ac, 6, sample, mo, 10, clip_sample=True, clip_sample_range=0.5)
    assert float(tight_x0.abs().max()) <= 0.5 + 1e-6, tight_x0


def check_random_against_oracle(solve):
    g = torch.Generator().manual_seed(20261029)
    types = ("epsilon", "sample", "v_prediction")
    for _ in range(20):
        num_train = int(torch.randint(4, 13, (1,), generator=g).item())
        ac = (torch.rand(num_train, generator=g) * 0.96 + 0.02).to(torch.float32)
        num_inf = int(torch.randint(1, num_train + 1, (1,), generator=g).item())
        timestep = int(torch.randint(0, num_train, (1,), generator=g).item())
        ptype = types[int(torch.randint(0, 3, (1,), generator=g).item())]
        clip = bool(torch.randint(0, 2, (1,), generator=g).item())
        alpha_one = bool(torch.randint(0, 2, (1,), generator=g).item())
        if ptype == "sample" and alpha_one and timestep < num_train // num_inf:
            alpha_one = False
        crange = float(torch.rand(1, generator=g).item()) * 1.5 + 0.1
        sample = torch.randn(2, 3, generator=g) * 2.0
        mo = torch.randn(2, 3, generator=g) * 2.0
        _check(
            solve, ac.tolist(), timestep, sample, mo, num_inf,
            prediction_type=ptype, clip_sample=clip, clip_sample_range=crange, set_alpha_to_one=alpha_one,
        )


def check_bad_input(solve):
    ac = torch.tensor([0.9, 0.8, 0.7, 0.6])
    sample = torch.tensor([0.1, 0.2])
    mo = torch.tensor([0.3, 0.4])
    bads = [
        lambda: solve([0.9, 0.8, 0.7, 0.6], 2, sample, mo, 4, clip_sample=False),
        lambda: solve(ac.reshape(2, 2), 2, sample, mo, 4, clip_sample=False),
        lambda: solve(torch.tensor([0, 1, 0, 1]), 2, sample, mo, 4, clip_sample=False),
        lambda: solve(torch.tensor([0.9, 0.0, 0.7, 0.6]), 2, sample, mo, 4, clip_sample=False),
        lambda: solve(torch.tensor([0.9, -0.1, 0.7, 0.6]), 2, sample, mo, 4, clip_sample=False),
        lambda: solve(torch.tensor([0.9, 1.5, 0.7, 0.6]), 2, sample, mo, 4, clip_sample=False),
        lambda: solve(torch.tensor([0.9, float("nan"), 0.7, 0.6]), 2, sample, mo, 4, clip_sample=False),
        lambda: solve(torch.tensor([]), 0, sample, mo, 1, clip_sample=False),
        lambda: solve(ac, 4, sample, mo, 4, clip_sample=False),
        lambda: solve(ac, -1, sample, mo, 4, clip_sample=False),
        lambda: solve(ac, 2.0, sample, mo, 4, clip_sample=False),
        lambda: solve(ac, True, sample, mo, 4, clip_sample=False),
        lambda: solve(ac, 2, sample, mo, 0, clip_sample=False),
        lambda: solve(ac, 2, sample, mo, 5, clip_sample=False),
        lambda: solve(ac, 2, [0.1, 0.2], mo, 4, clip_sample=False),
        lambda: solve(ac, 2, sample, torch.tensor([0.3]), 4, clip_sample=False),
        lambda: solve(ac, 2, sample, torch.tensor([float("inf"), 0.0]), 4, clip_sample=False),
        lambda: solve(ac, 2, sample, mo, 4, prediction_type="score", clip_sample=False),
        lambda: solve(ac, 2, sample, mo, 4, clip_sample=False, clip_sample_range=0.0),
        lambda: solve(ac, 2, sample, mo, 4, clip_sample=False, clip_sample_range=float("nan")),
        lambda: solve(ac, 2, sample, mo, 4, clip_sample="yes"),
        lambda: solve(ac, 2, sample, mo, 4, clip_sample=False, set_alpha_to_one=1),
    ]
    for fn in bads:
        try:
            fn()
        except ValueError:
            pass
        else:
            raise AssertionError("非法输入应抛 ValueError")
