import math

import torch


X0 = torch.tensor([[0.8, -0.3], [1.2, 0.1], [0.0, 1.5]], dtype=torch.float64)
F1 = torch.tensor([[-0.5, 1.0], [0.25, -0.75], [1.0, 0.5]], dtype=torch.float64)
F2 = torch.tensor([[0.3, -0.2], [-0.6, 0.9], [0.15, 0.45]], dtype=torch.float64)
Z = torch.tensor([[1.0, -1.0], [0.5, 0.0], [-0.25, 0.75]], dtype=torch.float64)


def _alpha_bar_fn(t, kind):
    if kind == "cosine":
        return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
    if kind == "laplace":
        lmb = -0.5 * math.copysign(1.0, 0.5 - t) * math.log(1.0 - 2.0 * math.fabs(0.5 - t) + 1e-6)
        snr = math.exp(lmb)
        return math.sqrt(snr / (1.0 + snr))
    return math.exp(t * -12.0)


def _oracle(x_init, f1, f2, z, train_t, kind, sd, marks, mb):
    """独立 oracle：用 sigma 表示系数（c_in = sqrt(1+sigma^2)/sqrt(sigma^2+sd^2)），
    与参考实现走 abar 倒数平方根那条路是两套代数式。"""
    betas = [
        min(1.0 - _alpha_bar_fn((i + 1) / train_t, kind) / _alpha_bar_fn(i / train_t, kind), mb)
        for i in range(train_t)
    ]
    abar = torch.tensor(betas, dtype=torch.float64)
    abar = torch.cumprod(1.0 - abar, dim=0)
    sigma = torch.sqrt(1.0 / abar - 1.0)
    den = sigma**2 + sd**2
    c_in = torch.sqrt(1.0 + sigma**2) / torch.sqrt(den)
    c_skip = torch.sqrt(1.0 + sigma**2) * (sd**2 / den)
    c_out = sigma * sd / torch.sqrt(den)
    t0, t1 = int(marks[0]), int(marks[1])
    x = x_init.to(torch.float64)
    scaled_first = x * c_in[t0]
    x0_first = c_out[t0] * f1.to(torch.float64) + c_skip[t0] * x
    x_second = abar[t1].sqrt() * x0_first + (1.0 - abar[t1]).sqrt() * z.to(torch.float64)
    scaled_second = x_second * c_in[t1]
    x_final = c_out[t1] * f2.to(torch.float64) + c_skip[t1] * x_second
    return scaled_first, x_second, scaled_second, x_final


def _round6(tensors):
    return [[[round(float(v), 6) for v in row] for row in t.tolist()] for t in tensors]


def check_example(solve):
    got = solve(X0, F1, F2, Z, 1024)
    want = [
        [[0.800156, -0.300059], [1.200235, 0.10002], [0.0, 1.500293]],
        [[0.540317, -0.362912], [0.448766, -0.262703], [0.172676, 0.715798]],
        [[0.680326, -0.456951], [0.565052, -0.330776], [0.217421, 0.901278]],
        [[0.284834, -0.19064], [-0.144342, 0.330657], [0.115307, 0.400921]],
    ]
    assert _round6(got) == want, _round6(got)
    for t in got:
        assert t.dtype == torch.float32 and tuple(t.shape) == (3, 2)


check_example.example = True


def check_hand_tiny_case(solve):
    # T=2, exp: beta 恒为 1 - exp(-6)，abar = [e^-6, e^-12]
    x = torch.ones(1, 1, dtype=torch.float64)
    zero = torch.zeros(1, 1, dtype=torch.float64)
    one = torch.ones(1, 1, dtype=torch.float64)
    sd = 0.5
    abar_t0, abar_t1 = math.exp(-12.0), math.exp(-6.0)  # marks = (1, 0)
    sig_t0 = math.sqrt(1.0 / abar_t0 - 1.0)
    sig_t1 = math.sqrt(1.0 / abar_t1 - 1.0)
    c_in_t0 = (1.0 / math.sqrt(abar_t0)) / math.sqrt(sig_t0**2 + sd**2)
    c_in_t1 = (1.0 / math.sqrt(abar_t1)) / math.sqrt(sig_t1**2 + sd**2)
    c_skip_t0 = (1.0 / math.sqrt(abar_t0)) * sd**2 / (sig_t0**2 + sd**2)
    c_skip_t1 = (1.0 / math.sqrt(abar_t1)) * sd**2 / (sig_t1**2 + sd**2)
    c_out_t1 = sig_t1 * sd / math.sqrt(sig_t1**2 + sd**2)

    got = solve(x, zero, zero, zero, 2, "exp", sd, (1, 0), 1.0)
    assert abs(float(got[0]) - c_in_t0) < 1e-6, float(got[0])
    x_second = math.sqrt(abar_t1) * c_skip_t0
    assert abs(float(got[1]) - x_second) < 1e-6, float(got[1])
    assert abs(float(got[2]) - x_second * c_in_t1) < 1e-6, float(got[2])
    assert abs(float(got[3]) - c_skip_t1 * x_second) < 1e-6, float(got[3])

    # noise=1、x=F1=0：x_second 只剩 sqrt(1 - abar[t1])
    got = solve(zero, zero, zero, one, 2, "exp", sd, (1, 0), 1.0)
    assert abs(float(got[1]) - math.sqrt(1.0 - abar_t1)) < 1e-6, float(got[1])
    assert abs(float(got[3]) - c_skip_t1 * math.sqrt(1.0 - abar_t1)) < 1e-6, float(got[3])
    # F2=1、x_second=0：x_final 就是 c_out[t1]
    got = solve(zero, zero, one, zero, 2, "exp", sd, (1, 0), 1.0)
    assert abs(float(got[3]) - c_out_t1) < 1e-6, float(got[3])


def check_three_transforms(solve):
    torch.manual_seed(0)
    tables = {}
    for kind in ("cosine", "laplace", "exp"):
        x = torch.randn(2, 3, dtype=torch.float64)
        f1 = torch.randn(2, 3, dtype=torch.float64)
        f2 = torch.randn(2, 3, dtype=torch.float64)
        z = torch.randn(2, 3, dtype=torch.float64)
        got = solve(x, f1, f2, z, 128, kind, 0.5, (100, 40), 0.999)
        want = _oracle(x, f1, f2, z, 128, kind, 0.5, (100, 40), 0.999)
        for g, w, tag in zip(got, want, ["scaled_first", "x_second", "scaled_second", "x_final"]):
            assert torch.allclose(g.to(torch.float64), w, rtol=1e-5, atol=1e-6), (kind, tag, g.tolist(), w.tolist())
        tables[kind] = got[0].clone()
    # 三种变换给出三套不同的 c_in
    assert not torch.allclose(tables["cosine"], tables["exp"], atol=1e-4)
    assert not torch.allclose(tables["laplace"], tables["exp"], atol=1e-4)
    assert not torch.allclose(tables["cosine"], tables["laplace"], atol=1e-4)


def check_max_beta_clamp(solve):
    x = torch.full((2, 2), 0.5, dtype=torch.float64)
    f1 = torch.full((2, 2), -0.25, dtype=torch.float64)
    f2 = torch.full((2, 2), 0.75, dtype=torch.float64)
    z = torch.full((2, 2), 0.125, dtype=torch.float64)
    for mb in (0.999, 0.5, 0.05):
        for kind in ("cosine", "laplace", "exp"):
            got = solve(x, f1, f2, z, 16, kind, 0.5, (12, 5), mb)
            want = _oracle(x, f1, f2, z, 16, kind, 0.5, (12, 5), mb)
            for g, w in zip(got, want):
                assert torch.allclose(g.to(torch.float64), w, rtol=1e-5, atol=1e-7), (kind, mb)
    # 截断真的会绑定：cosine 尾部原始 beta 已达 1.0，压到 0.05 后 x_second 显著变大
    tight = solve(x, f1, f2, z, 16, "cosine", 0.5, (12, 5), 0.05)
    loose = solve(x, f1, f2, z, 16, "cosine", 0.5, (12, 5), 0.999)
    assert not torch.allclose(tight[1], loose[1], atol=1e-3)
    # max_beta 放到 1.0 时 exp 仍安全（beta 恒定），cosine 尾步会把 alpha_bar 打到 0
    assert solve(x, f1, f2, z, 16, "exp", 0.5, (12, 5), 1.0)[3].dtype == torch.float32
    try:
        solve(x, f1, f2, z, 16, "cosine", 0.5, (12, 5), 1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("cosine + max_beta=1.0 应因 alpha_bar 触零而抛 ValueError")


def check_renoise_uses_next_timestep(solve):
    torch.manual_seed(3)
    t1 = 77
    x = torch.randn(4, 3, dtype=torch.float64)
    f1 = torch.randn(4, 3, dtype=torch.float64)
    f2 = torch.randn(4, 3, dtype=torch.float64)
    za = torch.randn(4, 3, dtype=torch.float64)
    zb = torch.randn(4, 3, dtype=torch.float64)
    zero = torch.zeros(4, 3, dtype=torch.float64)
    unit = torch.ones(4, 3, dtype=torch.float64)

    def run(xa, fa, fb, nz, mark):
        return solve(xa, fa, fb, nz, 256, "cosine", 0.5, mark, 0.999)

    # F1=0、x=0 时 x_second 只剩 sqrt(1 - abar[t1]) * noise
    probe = run(zero, zero, zero, unit, (200, t1))
    k = float(probe[1][0, 0])
    assert 0.0 < k < 1.0, k
    got_a, got_b = run(x, f1, f2, za, (200, t1)), run(x, f1, f2, zb, (200, t1))
    slope_second = ((got_a[1] - got_b[1]) / (za - zb)).to(torch.float64)
    assert torch.allclose(slope_second, torch.full_like(slope_second, k), rtol=1e-5), slope_second.tolist()
    # 末步不再加噪：x_final 只能通过 x_second 沾染 noise
    c_skip = float(probe[3][0, 0]) / k
    slope_final = ((got_a[3] - got_b[3]) / (za - zb)).to(torch.float64)
    assert torch.allclose(slope_final, torch.full_like(slope_final, k * c_skip), rtol=1e-5), slope_final.tolist()
    # 重加噪系数跟着 t1 走，跟着当前步走就露馅
    other = run(x, f1, f2, za, (200, 12))
    assert not torch.allclose(got_a[1], other[1], atol=1e-3)


def check_last_step_is_plain_denoise(solve):
    torch.manual_seed(4)
    x = torch.randn(3, 5, dtype=torch.float64)
    f1 = torch.randn(3, 5, dtype=torch.float64)
    z = torch.randn(3, 5, dtype=torch.float64)
    zero = torch.zeros(3, 5, dtype=torch.float64)
    got = solve(x, f1, zero, z, 512, "laplace", 0.5, (400, 90), 0.999)
    # F2 = 0 -> x_final 与 x_second 逐元素成同一比例 c_skip[t1]
    k = float(got[3][0, 0]) / float(got[1][0, 0])
    assert 0.0 < k < 1.0, k
    assert torch.allclose(got[3].to(torch.float64), k * got[1].to(torch.float64), rtol=1e-4, atol=1e-7), got[3].tolist()


def check_custom_marks(solve):
    torch.manual_seed(5)
    x = torch.randn(2, 4, dtype=torch.float64)
    f1 = torch.randn(2, 4, dtype=torch.float64)
    f2 = torch.randn(2, 4, dtype=torch.float64)
    z = torch.randn(2, 4, dtype=torch.float64)
    for marks in ((1023, 0), (1, 0), (1008, 512), (600, 599), (300, 1)):
        got = solve(x, f1, f2, z, 1024, "cosine", 0.5, marks, 0.999)
        want = _oracle(x, f1, f2, z, 1024, "cosine", 0.5, marks, 0.999)
        for g, w in zip(got, want):
            assert torch.allclose(g.to(torch.float64), w, rtol=1e-5, atol=1e-6), (marks, g.tolist())


def check_against_oracle(solve):
    torch.manual_seed(11)
    for trial in range(60):
        train_t = [2, 8, 64, 300, 1024][trial % 5]
        kind = ["cosine", "laplace", "exp"][trial % 3]
        sd = [0.5, 1.0, 0.25][trial % 3]
        mb = [0.999, 0.5, 1.0][trial % 3]
        hi = int(torch.randint(max(1, train_t // 2), train_t, (1,)))
        lo = int(torch.randint(0, max(1, train_t // 2), (1,)))
        x = torch.randn(3, 2, dtype=torch.float64)
        f1 = torch.randn(3, 2, dtype=torch.float64)
        f2 = torch.randn(3, 2, dtype=torch.float64)
        z = torch.randn(3, 2, dtype=torch.float64)
        got = solve(x, f1, f2, z, train_t, kind, sd, (hi, lo), mb)
        want = _oracle(x, f1, f2, z, train_t, kind, sd, (hi, lo), mb)
        for g, w, tag in zip(got, want, ["scaled_first", "x_second", "scaled_second", "x_final"]):
            assert torch.allclose(g.to(torch.float64), w, rtol=2e-4, atol=2e-5), (trial, tag, train_t, kind, sd, mb, hi, lo)


def check_dtypes_and_shapes(solve):
    c1, c2, c3, c4 = solve(X0.to(torch.float32), F1.to(torch.float32), F2.to(torch.float32), Z.to(torch.float32), 1024)
    for t in (c1, c2, c3, c4):
        assert t.dtype == torch.float32 and tuple(t.shape) == (3, 2) and torch.isfinite(t).all()
    big = torch.randn(2, 3, 4, 5, dtype=torch.float32)
    out = solve(big, big, big, big, 256, "exp", 1.0, (200, 8), 0.999)
    for t in out:
        assert t.dtype == torch.float32 and tuple(t.shape) == (2, 3, 4, 5)
    flat = torch.randn(4, dtype=torch.float64)
    out = solve(flat, flat, flat, flat, 64, "laplace", 0.5, (50, 3), 0.999)
    assert tuple(out[0].shape) == (4,)
    # c_in 在两个 mark 上确实不同 -> scaled_first != scaled_second 的比例
    assert float(out[0][0]) != float(out[2][0])


def _poison(t, value):
    out = t.clone()
    out[0, 0] = value
    return out


def check_invalid_inputs(solve):
    bad = [
        ("x_init 非张量", lambda: solve([[0.1, 0.2]], F1[:1, :2], F1[:1, :2], F1[:1, :2], 16)),
        ("x_init 0 维", lambda: solve(torch.tensor(1.0), None, None, None, 16)),
        ("形状不一致", lambda: solve(X0, F1[:2], F2, Z, 1024)),
        ("含 NaN", lambda: solve(_poison(X0, float("nan")), F1, F2, Z, 1024)),
        ("含 inf", lambda: solve(X0, F1, _poison(F2, float("inf")), Z, 1024)),
        ("noise 非张量", lambda: solve(X0, F1, F2, [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], 1024)),
        ("T 为浮点", lambda: solve(X0, F1, F2, Z, 1024.0)),
        ("T 为 bool", lambda: solve(X0, F1, F2, Z, True)),
        ("T = 1", lambda: solve(X0, F1, F2, Z, 1)),
        ("T = 0", lambda: solve(X0, F1, F2, Z, 0)),
        ("变换名不认识", lambda: solve(X0, F1, F2, Z, 1024, "sigmoid")),
        ("变换名非字符串", lambda: solve(X0, F1, F2, Z, 1024, 1)),
        ("sigma_data = 0", lambda: solve(X0, F1, F2, Z, 1024, "cosine", 0.0)),
        ("sigma_data 为负", lambda: solve(X0, F1, F2, Z, 1024, "cosine", -0.5)),
        ("sigma_data 为 NaN", lambda: solve(X0, F1, F2, Z, 1024, "cosine", float("nan"))),
        ("max_beta = 0", lambda: solve(X0, F1, F2, Z, 1024, "cosine", 0.5, (1008, 512), 0.0)),
        ("max_beta > 1", lambda: solve(X0, F1, F2, Z, 1024, "cosine", 0.5, (1008, 512), 1.5)),
        ("timesteps 长度为 1", lambda: solve(X0, F1, F2, Z, 1024, "cosine", 0.5, (1008,))),
        ("timesteps 长度为 3", lambda: solve(X0, F1, F2, Z, 1024, "cosine", 0.5, (1008, 512, 3), 0.999)),
        ("timesteps 为字符串", lambda: solve(X0, F1, F2, Z, 1024, "cosine", 0.5, "1008", 0.999)),
        ("timesteps 元素为浮点", lambda: solve(X0, F1, F2, Z, 1024, "cosine", 0.5, (1008.0, 512), 0.999)),
        ("timesteps 元素越界", lambda: solve(X0, F1, F2, Z, 1024, "cosine", 0.5, (1024, 512), 0.999)),
        ("timesteps 递增", lambda: solve(X0, F1, F2, Z, 1024, "cosine", 0.5, (512, 1008), 0.999)),
        ("timesteps 相等", lambda: solve(X0, F1, F2, Z, 1024, "cosine", 0.5, (512, 512), 0.999)),
    ]
    for name, fn in bad:
        try:
            fn()
        except ValueError:
            continue
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(f"{name}: 抛了 {type(exc).__name__}，应为 ValueError") from exc
        raise AssertionError(f"{name}: 未抛 ValueError")
