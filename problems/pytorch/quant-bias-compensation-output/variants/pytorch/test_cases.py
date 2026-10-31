"""Independent pure-Python oracle (scalar loops over channel/non-channel index tuples) plus targeted checks."""

import math

import torch

KEYS = ("channel_axis_norm", "bias_shape", "mean_axis", "bias_updates", "bias_after",
        "bias", "corrected_output", "residual_mse")


def _normalize_axes(channel_axis, d):
    if isinstance(channel_axis, (tuple, list)):
        raw = tuple(channel_axis)
    elif isinstance(channel_axis, int) and not isinstance(channel_axis, bool):
        raw = (channel_axis,)
    else:
        raise ValueError("channel_axis must be an int or a tuple/list of ints")
    if not raw:
        raise ValueError("channel_axis must not be empty")
    if any(not isinstance(a, int) or isinstance(a, bool) for a in raw):
        raise ValueError("channel_axis entries must be ints")
    norm = tuple(a + d if a < 0 else a for a in raw)
    if any(a < 0 or a >= d for a in norm):
        raise ValueError("channel_axis out of range")
    if len(set(norm)) != len(norm):
        raise ValueError("channel_axis entries must be distinct")
    return norm


def _iter_indices(sizes):
    """Row-major multi-indices of a shape given as a tuple of ints."""
    total = 1
    for s in sizes:
        total *= s
    for flat in range(total):
        idx = []
        rest = flat
        for s in reversed(sizes):
            idx.append(rest % s)
            rest //= s
        yield tuple(reversed(idx))


def _at(frame, pos):
    """Index a nested list with a multi-index (pos may be any length >= 1)."""
    node = frame
    for p in pos:
        node = node[p]
    return node


def _linear(sizes, idx):
    flat = 0
    for s, v in zip(sizes, idx):
        flat = flat * s + v
    return flat


def _oracle(float_batches, quant_batches, channel_axis, init_bias=None, post_bias=True,
            channel_size=None):
    """纯 Python 标量实现：逐 (t, 通道多重索引) 手算均值与流式平均，完全不碰张量归约。"""
    Fdim = tuple(float_batches.shape)
    if Fdim != tuple(quant_batches.shape):
        raise ValueError("shape mismatch")
    if float_batches.dtype != quant_batches.dtype:
        raise ValueError("dtype mismatch")
    if len(Fdim) < 3:
        raise ValueError("single-forward rank must be >= 2")
    if Fdim[0] == 0:
        raise ValueError("at least one calibration output pair is required")
    if float_batches.dtype not in (torch.float32, torch.float64):
        raise ValueError("float_batches must be float32 or float64")

    d = len(Fdim) - 1
    norm = _normalize_axes(channel_axis, d)
    spatial = Fdim[1:]
    bias_shape = tuple(spatial[i] if i in norm else 1 for i in range(d))
    mean_axis = tuple(i for i in range(d) if i not in norm)
    channels = 1
    for size in bias_shape:
        channels *= size
    if channel_size is not None and int(channel_size) != channels:
        raise ValueError("channel_size does not match prod(bias_shape)")

    fl = float_batches.detach().to(torch.float64).contiguous().tolist()
    ql = quant_batches.detach().to(torch.float64).contiguous().tolist()
    if init_bias is None:
        bias = [0.0] * channels
    else:
        flat = init_bias.detach().to(torch.float64).contiguous().reshape(-1).tolist()
        if len(flat) != channels:
            raise ValueError("init_bias length does not match prod(bias_shape)")
        bias = list(flat)

    chan_axes = sorted(norm)
    chan_sizes = tuple(spatial[i] for i in chan_axes)
    other_sizes = tuple(spatial[i] for i in mean_axis)
    T = len(fl)
    updates = []
    after = []
    for t in range(T):
        u = []
        for ci in _iter_indices(chan_sizes):
            acc = 0.0
            count = 0
            for oi in _iter_indices(other_sizes) if other_sizes else [()]:
                pos = []
                left_c = list(ci)
                left_o = list(oi)
                for i in range(d):
                    if i in norm:
                        pos.append(left_c.pop(0))
                    else:
                        pos.append(left_o.pop(0))
                delta = _at(fl[t], pos) - _at(ql[t], pos)
                if post_bias:
                    slot = [pos[i] for i in chan_axes]
                    delta += bias[_linear(chan_sizes, slot)]
                acc += delta
                count += 1
            u.append(acc / count)
        updates.append(list(u))
        bias = [(bias[k] * t + u[k]) / (t + 1) for k in range(channels)]
        after.append(list(bias))

    corrected = []
    se = 0.0
    n = 0
    for t in range(T):
        for ci in _iter_indices(spatial):
            linear = _linear(chan_sizes, [ci[i] for i in chan_axes])
            qv = _at(ql[t], list(ci))
            cv = qv + bias[linear]
            corrected.append(cv)
            se += (_at(fl[t], list(ci)) - cv) ** 2
            n += 1
    mse = se / n

    return {
        "channel_axis_norm": norm,
        "bias_shape": bias_shape,
        "mean_axis": mean_axis,
        "bias_updates": torch.tensor(updates, dtype=torch.float64).reshape(T, channels),
        "bias_after": torch.tensor(after, dtype=torch.float64).reshape(T, channels),
        "bias": torch.tensor(bias, dtype=torch.float64).reshape(channels),
        "corrected_output": torch.tensor(corrected, dtype=torch.float64).reshape((T,) + spatial),
        "residual_mse": torch.tensor(mse, dtype=torch.float64),
    }


def _case(T=2, B=2, C=3, S=2, axis=1, seed=0, dtype=torch.float32, channel_axis=None):
    generator = torch.Generator().manual_seed(seed)
    shape = (T, B, C, S)
    F = torch.randn(shape, generator=generator, dtype=torch.float64).to(dtype) * 0.9
    Q = F + torch.randn(shape, generator=generator, dtype=torch.float64).to(dtype) * 0.1 - 0.02
    use_axis = axis if channel_axis is None else channel_axis
    return (F, Q, use_axis)


def _example():
    F = torch.tensor([[[[0.10, 0.20], [0.30, 0.40], [0.50, 0.60]],
                       [[0.70, 0.80], [0.90, 1.00], [0.11, 0.22]]],
                      [[[0.12, 0.18], [0.34, 0.36], [0.48, 0.62]],
                       [[0.68, 0.82], [0.88, 1.02], [0.09, 0.25]]]], dtype=torch.float64)
    Q = torch.tensor([[[[0.05, 0.15], [0.25, 0.35], [0.45, 0.55]],
                       [[0.60, 0.75], [0.85, 0.95], [0.05, 0.20]]],
                      [[[0.06, 0.14], [0.30, 0.30], [0.44, 0.60]],
                       [[0.62, 0.78], [0.80, 1.00], [0.10, 0.24]]]], dtype=torch.float64)
    return (F, Q, 1)


def _example_two():
    F = torch.tensor([[[0.10, 0.20, 0.30], [0.40, 0.50, 0.60]],
                      [[0.20, 0.10, 0.40], [0.50, 0.30, 0.70]]], dtype=torch.float64)
    Q = torch.tensor([[[0.00, 0.25, 0.35], [0.45, 0.55, 0.50]],
                      [[0.15, 0.15, 0.45], [0.55, 0.35, 0.65]]], dtype=torch.float64)
    return (F, Q, (0, 1))


def _close(a, b):
    tol = 3e-5 if a.dtype == torch.float32 else 2e-9
    torch.testing.assert_close(a, b, rtol=tol, atol=tol)


def _close_loose(a, b):
    """手算期望值来自 float32 回读后的张量，容差按 float32 走。"""
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)


def _split(args, kwargs=None):
    """把 kwargs 里的 channel_axis 归位到第三个位置参数，避免重复传参。"""
    F, Q, axis = args
    kw = dict(kwargs or {})
    axis = kw.pop("channel_axis", axis)
    return F, Q, axis, kw


def _check_structure(solve, args, kwargs=None):
    F, Q, axis, kw = _split(args, kwargs)
    before_f, before_q = F.clone(), Q.clone()
    got = solve(F, Q, axis, **kw)
    assert isinstance(got, dict) and set(got) == set(KEYS), f"键不对：{sorted(got)}"
    assert torch.equal(F, before_f) and torch.equal(Q, before_q), "输入被就地修改了"
    T = int(F.shape[0])
    d = F.dim() - 1
    C = int(got["bias"].numel())
    assert got["channel_axis_norm"] == tuple(got["channel_axis_norm"])
    assert len(got["bias_shape"]) == d
    assert got["bias"].shape == (C,)
    assert got["bias_updates"].shape == (T, C)
    assert got["bias_after"].shape == (T, C)
    assert got["corrected_output"].shape == F.shape
    assert got["residual_mse"].dim() == 0
    for key in ("bias_updates", "bias_after", "bias", "corrected_output", "residual_mse"):
        assert got[key].dtype == F.dtype, f"{key} dtype {got[key].dtype} != {F.dtype}"
        assert torch.isfinite(got[key]).all(), f"{key} 出现非有限值"
    prod = 1
    for size in got["bias_shape"]:
        prod *= size
    assert prod == C, "prod(bias_shape) 与偏置长度不一致"
    assert sorted(got["mean_axis"]) == list(got["mean_axis"]), "mean_axis 必须升序"
    assert set(got["mean_axis"]).isdisjoint(set(got["channel_axis_norm"]))
    return got


def _verify(solve, args, kwargs=None):
    got = _check_structure(solve, args, kwargs)
    F, Q, axis, kw = _split(args, kwargs)
    want = _oracle(F, Q, axis, **kw)
    want = {k: (v.to(F.dtype) if isinstance(v, torch.Tensor) else v)
            for k, v in want.items()}
    for key in ("channel_axis_norm", "bias_shape", "mean_axis"):
        assert got[key] == want[key], f"{key}：{got[key]} vs {want[key]}"
    for key in ("bias_updates", "bias_after", "bias", "corrected_output", "residual_mse"):
        _close(got[key], want[key])
    return got


def check_example(solve):
    """题面示例一：(T,B,C,S)=(2,2,3,2)、channel_axis=1 的全部八个键。"""
    got = _verify(solve, _example())
    assert got["channel_axis_norm"] == (1,)
    assert got["bias_shape"] == (1, 3, 1)
    assert got["mean_axis"] == (0, 2)
    expect = torch.tensor([[0.0625, 0.05, 0.045], [0.1125, 0.1, 0.06]], dtype=torch.float64)
    _close(got["bias_updates"].double(), expect)
    _close(got["bias"].double(), torch.tensor([0.0875, 0.075, 0.0525], dtype=torch.float64))
    assert abs(float(got["residual_mse"]) - 0.0010604166666666663) < 1e-12
    assert abs(float(got["corrected_output"][0][0][0][0]) - 0.1375) < 1e-12


check_example.example = True


def check_example_all_channel_axes(solve):
    """题面示例二：channel_axis=(0,1) 让 mean_axis 为空，此时不取均值、逐元素成通道。"""
    got = _verify(solve, _example_two())
    assert got["mean_axis"] == ()
    assert got["bias_shape"] == (2, 3)
    assert got["bias"].numel() == 6
    assert abs(float(got["residual_mse"]) - 0.0014583333333333334) < 1e-12
    _close(got["bias_updates"][0].double(),
           torch.tensor([0.1, -0.05, -0.05, -0.05, -0.05, 0.1], dtype=torch.float64))
    _close(got["bias"].double(),
           torch.tensor([0.125, -0.075, -0.075, -0.075, -0.075, 0.125], dtype=torch.float64))


check_example_all_channel_axes.example = True


def check_negative_axis_normalization(solve):
    """负轴按单次前向的秩 d = dim-1 归一化，而不是按 dim()。"""
    F, Q, _ = _example()
    pos = solve(F, Q, 1)
    neg = solve(F, Q, -2)
    assert pos["channel_axis_norm"] == neg["channel_axis_norm"] == (1,)
    for key in ("bias_updates", "bias_after", "bias", "residual_mse"):
        _close(pos[key], neg[key])
    edge = solve(F, Q, 0)
    assert edge["channel_axis_norm"] == (0,) and edge["bias_shape"] == (2, 1, 1)


def check_mean_axis_empty_flattens(solve):
    """所有轴都是通道轴时，偏置长度等于单次前向的元素个数。"""
    args = _case(T=3, seed=11)
    F, Q, _ = args
    d = F.dim() - 1
    got = _verify(solve, args, {"channel_axis": tuple(range(d))})
    assert got["mean_axis"] == ()
    assert got["bias"].numel() == F[0].numel()
    assert torch.allclose(got["bias_updates"][0].reshape(F[0].shape),
                          (F[0] - Q[0]).to(F.dtype), rtol=1e-5, atol=1e-5)


def check_streaming_weights(solve):
    """b_{t+1} = (b_t * t + u_t)/(t+1)：第一次覆盖初值，其后权重都是 1/(t+1)。"""
    args = _case(T=5, seed=3)
    got = _verify(solve, args)
    u = got["bias_updates"].double()
    a = got["bias_after"].double()
    _close_loose(a[0], u[0])
    for t in range(1, 5):
        _close_loose(a[t], (a[t - 1] * t + u[t]) / (t + 1))
    _close_loose(a[4], u.mean(dim=0))


def check_init_bias_shifts_only_with_post_bias(solve):
    """post_bias=False 时 b_1 的权重是 0，初值全程无效；post_bias=True 时它随 delta 回流，整条轨迹平移 init_bias。"""
    init = torch.full((3,), 7.0, dtype=torch.float64)
    two = _case(T=2, seed=5)
    plain = _verify(solve, two, {"post_bias": False})
    seeded = _verify(solve, two, {"post_bias": False, "init_bias": init})
    for key in ("bias_updates", "bias_after", "bias", "corrected_output", "residual_mse"):
        _close(plain[key], seeded[key])
    one = _case(T=1, seed=5)
    p1 = _verify(solve, one, {"post_bias": True})
    q1 = _verify(solve, one, {"post_bias": True, "init_bias": init})
    off = torch.full((3,), 7.0, dtype=p1["bias"].dtype)
    _close(q1["bias"], p1["bias"] + off)
    _close(q1["bias_updates"], p1["bias_updates"] + off)
    p2 = _verify(solve, two, {"post_bias": True})
    q2 = _verify(solve, two, {"post_bias": True, "init_bias": init})
    _close(q2["bias"], p2["bias"] + off)
    _close(q2["bias_after"], p2["bias_after"] + off)
    _close_loose(q2["bias"].double() - p2["bias"].double(),
                 torch.full((3,), 7.0, dtype=torch.float64))


def check_post_bias_uses_previous_bias(solve):
    """post_bias=True 加回的是上一次的偏置；T=1 时与 False 等价，T>=2 时不再等价。"""
    one = _case(T=1, seed=7)
    t_true = _verify(solve, one, {"post_bias": True})
    t_false = _verify(solve, one, {"post_bias": False})
    _close(t_true["bias"], t_false["bias"])
    two = _case(T=2, seed=7)
    f_true = solve(*two, post_bias=True)
    f_false = solve(*two, post_bias=False)
    assert not torch.allclose(f_true["bias"], f_false["bias"], rtol=1e-4, atol=1e-4)
    manual = (f_false["bias_updates"][0].double()
              + f_false["bias_updates"][1].double()
              + f_false["bias_updates"][0].double()) / 2.0
    _close_loose(f_true["bias"].double(), manual)


def check_batch_axis_is_averaged_away(solve):
    """沿单次前向的 batch 轴复制一份：通道均值与整条偏置轨迹都不变。"""
    base = _case(T=3, seed=9)
    got = _verify(solve, base)
    F, Q, axis = base
    tiled_F = torch.cat([F, F], dim=1)
    tiled_Q = torch.cat([Q, Q], dim=1)
    dup = _verify(solve, (tiled_F, tiled_Q, axis))
    assert dup["bias_shape"] == got["bias_shape"] == (1, 3, 1)
    _close(dup["bias"], got["bias"])
    _close(dup["bias_updates"], got["bias_updates"])


def check_identical_outputs_zero_bias(solve):
    """F == Q 且初值为零时偏置恒为 0，corrected_output 回到 Q，MSE 为 0。"""
    args = _case(T=4, seed=13)
    F = args[0]
    got = _verify(solve, (F, F.clone(), args[2]))
    assert torch.equal(got["bias"], torch.zeros_like(got["bias"]))
    assert torch.equal(got["bias_updates"], torch.zeros_like(got["bias_updates"]))
    assert torch.equal(got["corrected_output"], F)
    assert float(got["residual_mse"]) == 0.0


def check_corrected_output_and_mse_consistency(solve):
    """corrected_output 用最终偏置广播；residual_mse 等于其逐元素平方误差均值。"""
    args = _case(T=3, seed=17)
    F, Q, axis = args
    got = _verify(solve, args)
    bias_shape = got["bias_shape"]
    b = got["bias"].reshape((1,) + bias_shape)
    _close(got["corrected_output"], Q + b)
    recomputed = ((F - got["corrected_output"]).double() ** 2).mean()
    _close(got["residual_mse"].double().reshape(()), recomputed.reshape(()))


def check_multi_channel_axes(solve):
    """两个通道轴：bias_shape 在两处填尺寸，mean_axis 只剩一个轴。"""
    args = _case(T=2, seed=19)
    F, Q, _ = args
    got = _verify(solve, args, {"channel_axis": (0, 2)})
    assert got["bias_shape"] == (2, 1, 2)
    assert got["mean_axis"] == (1,)
    assert got["bias"].numel() == 4
    assert got["channel_axis_norm"] == (0, 2)


def check_channel_size_validation(solve):
    """channel_size 与 prod(bias_shape) 相符时结果不变，不符时抛 ValueError。"""
    args = _example()
    ok = _verify(solve, args, {"channel_size": 3})
    assert ok["bias"].numel() == 3
    for bad in (0, 2, 4, 6):
        try:
            solve(*args, channel_size=bad)
        except ValueError:
            continue
        raise AssertionError(f"channel_size={bad} 应当抛 ValueError")


def check_dtype_sweep(solve):
    """float32 与 float64 输入下的结果一致到容差，输出 dtype 跟随输入。"""
    f64 = _case(T=3, seed=23, dtype=torch.float64)
    f32 = (f64[0].float(), f64[1].float(), f64[2])
    got64 = _verify(solve, f64)
    got32 = _verify(solve, f32)
    assert got32["bias"].dtype == torch.float32 and got64["bias"].dtype == torch.float64
    torch.testing.assert_close(got32["bias"].double(), got64["bias"], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(got32["residual_mse"].double(), got64["residual_mse"],
                               rtol=1e-5, atol=1e-5)


def check_non_contiguous_input(solve):
    """permute 得到的非连续输入：结果与连续副本、也与原布局下的同一通道轴一致。"""
    F, Q, _ = _case(T=2, B=3, C=4, S=2, seed=29)
    Fp = F.permute(0, 2, 1, 3)
    Qp = Q.permute(0, 2, 1, 3)
    assert not Fp.is_contiguous()
    got = _verify(solve, (Fp, Qp, 0))
    assert got["bias_shape"] == (4, 1, 1)
    ref = solve(Fp.contiguous(), Qp.contiguous(), 0)
    for key in ("bias_updates", "bias_after", "bias", "corrected_output", "residual_mse"):
        _close(got[key], ref[key])
    orig = solve(F, Q, 1)
    _close(got["bias"], orig["bias"])


def check_inputs_not_mutated(solve):
    """两个输出张量与 init_bias 都不被就地修改。"""
    args = _case(T=3, seed=31)
    bias = torch.arange(3, dtype=torch.float64)
    before = (args[0].clone(), args[1].clone(), bias.clone())
    solve(*args, init_bias=bias)
    assert torch.equal(args[0], before[0]) and torch.equal(args[1], before[1])
    assert torch.equal(bias, before[2])


def check_value_errors(solve):
    """形状/dtype/秩/T=0/空轴/越界轴/重复轴/非整数/init_bias 长度不符一律 ValueError。"""
    F, Q, axis = _example()
    bad_calls = [
        (F[:0], Q[:0], axis, {}),
        (F[0][0], Q[0][0], 0, {}),
        (F, Q[:, :, :, :1], axis, {}),
        (F, Q.float(), axis, {}),
        (F, Q, (), {}),
        (F, Q, 3, {}),
        (F, Q, -4, {}),
        (F, Q, (1, 1), {}),
        (F, Q, (0, 1.5), {}),
        (F, Q, "1", {}),
        (F, Q, axis, {"channel_size": 5}),
        (F, Q, axis, {"init_bias": torch.zeros(4, dtype=torch.float64)}),
    ]
    for args_f, args_q, args_axis, kwargs in bad_calls:
        try:
            solve(args_f, args_q, args_axis, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"未抛 ValueError：shape={tuple(args_f.shape)} axis={args_axis} {kwargs}")


def check_random_against_oracle(solve):
    """80 组随机实例（多种秩、通道轴组合、post_bias / init_bias）与纯 Python oracle 一致。"""
    generator = torch.Generator().manual_seed(20261101)
    for case in range(80):
        rank = int(torch.randint(2, 5, (1,), generator=generator))
        shape = [int(torch.randint(1, 4, (1,), generator=generator)) for _ in range(rank)]
        shape[0] = max(1, shape[0])
        T = int(torch.randint(1, 5, (1,), generator=generator))
        full = [T] + shape
        d = len(shape)
        n_ch = int(torch.randint(1, d + 1, (1,), generator=generator))
        axes = sorted(torch.randperm(d, generator=generator)[:n_ch].tolist())
        if case % 4 == 0:
            axes = [-1] if axes == [0] else [axes[-1] - d]
        F = torch.randn(*full, generator=generator, dtype=torch.float64)
        Q = F + torch.randn(*full, generator=generator, dtype=torch.float64) * 0.2
        if case % 5 == 0:
            F = torch.zeros(*full, dtype=torch.float64)
        if case % 6 == 0:
            Q = Q - Q
        post = bool(case % 3)
        init = None
        if case % 7 == 3:
            norm_axes = {a + d if a < 0 else a for a in axes}
            prod = 1
            for i, s in enumerate(shape):
                if i in norm_axes:
                    prod *= s
            init = torch.randn(prod, generator=generator, dtype=torch.float64)
        args = (F.float(), Q.float(), tuple(axes))
        kwargs = {"post_bias": post, "init_bias": init}
        got = solve(*args, **kwargs)
        want = _oracle(*args, **kwargs)
        tag = (case, full, axes, post)
        assert got["channel_axis_norm"] == want["channel_axis_norm"], f"{tag}: 轴不符"
        assert got["bias_shape"] == want["bias_shape"], f"{tag}: bias_shape 不符"
        assert got["mean_axis"] == want["mean_axis"], f"{tag}: mean_axis 不符"
        for key in ("bias_updates", "bias_after", "bias", "corrected_output", "residual_mse"):
            _close(got[key], want[key].to(torch.float32))
