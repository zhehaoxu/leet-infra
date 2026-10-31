import math

import torch


BETAS64 = [0.05, 0.08, 0.12, 0.18, 0.25, 0.33, 0.40, 0.50]

X = torch.tensor([[0.4, -0.7], [1.6, 0.2], [-0.3, 0.9]], dtype=torch.float32)
EPS = torch.tensor([[0.5, -0.25], [0.75, 0.1], [-0.6, 0.4]], dtype=torch.float32)
ORIG = torch.tensor([[-0.8, 0.6], [0.3, -0.5], [0.9, 0.2]], dtype=torch.float32)
MASK = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
Z = torch.tensor([[0.7, -0.4], [0.2, 0.9], [-0.5, 0.3]], dtype=torch.float32)
UN = torch.tensor(
    [[[0.1, 0.2], [-0.3, 0.4], [0.5, -0.1]], [[-0.2, 0.3], [0.4, -0.5], [0.1, 0.6]]],
    dtype=torch.float32,
)


def _betas_tensor():
    return torch.tensor(BETAS64, dtype=torch.float32)


def _abar_table(betas):
    """纯 Python 前缀积：abar[t] = prod_{j<=t} (1 - beta_j)。"""
    out = []
    acc = 1.0
    for b in betas:
        acc *= 1.0 - b
        out.append(acc)
    return out


def _coeffs(betas, t, n_in):
    """返回 (n, abar_t, abar_prev, var, eta_max)。"""
    abar = _abar_table(betas)
    T = len(betas)
    n = T // n_in
    ab_t = abar[t]
    ab_p = abar[t - n] if t - n >= 0 else 1.0
    var = ((1.0 - ab_p) / (1.0 - ab_t)) * (1.0 - ab_t / ab_p)
    if ab_p >= 1.0:
        cap = math.inf
    else:
        cap = math.sqrt((1.0 - ab_t) / (1.0 - ab_t / ab_p))
    return n, ab_t, ab_p, var, cap


def _oracle(sample, eps, orig, mask, noise, betas, undo_noise, t, n_in, eta, clip):
    """独立 oracle：x̂₀ 拆成 x 与 ε 两个系数、回跳用闭式权重，与参考实现不同源。"""
    n, ab_t, ab_p, var, _ = _coeffs(betas, t, n_in)
    cx = 1.0 / math.sqrt(ab_t)
    ce = math.sqrt((1.0 - ab_t) / ab_t)
    std = eta * math.sqrt(var)
    dcoef = math.sqrt((1.0 - ab_p) - std * std)
    add = std if (t > 0 and eta > 0) else 0.0
    kp = math.sqrt(1.0 - ab_p)
    sp = math.sqrt(ab_p)

    prev, x0_out = [], []
    for row_x, row_e, row_o, row_m, row_z in zip(
        sample.tolist(), eps.tolist(), orig.tolist(), mask.tolist(), noise.tolist()
    ):
        p_row, x_row = [], []
        for xv, ev, ov, mv, zv in zip(row_x, row_e, row_o, row_m, row_z):
            est = cx * xv - ce * ev
            if clip:
                est = min(1.0, max(-1.0, est))
            unknown = sp * est + dcoef * ev + add * zv
            known = sp * ov + kp * zv
            p_row.append(mv * known + (1.0 - mv) * unknown)
            x_row.append(est)
        prev.append(p_row)
        x0_out.append(x_row)

    # 回跳闭式：r = prod_i alpha_{t+i}；undo_noise[i] 的系数 = sqrt(beta_{t+i} * prod_{k>i} alpha_{t+k})
    alphas = [1.0 - b for b in betas[t : t + n]]
    r = 1.0
    for a in alphas:
        r *= a
    weights = []
    for i in range(n):
        tail = 1.0
        for a in alphas[i + 1 :]:
            tail *= a
        weights.append(math.sqrt(betas[t + i] * tail))
    rows = sample.shape[0]
    cols = sample.shape[1]
    flat_x = [v for row in sample.tolist() for v in row]
    flat_un = [[v for row in u.tolist() for v in row] for u in undo_noise]
    flat_out = []
    for idx in range(len(flat_x)):
        acc = math.sqrt(r) * flat_x[idx]
        for i in range(n):
            acc += weights[i] * flat_un[i][idx]
        flat_out.append(acc)
    undone = [flat_out[i * cols : (i + 1) * cols] for i in range(rows)]
    return prev, x0_out, undone


def _flat(x):
    if isinstance(x, torch.Tensor):
        return x.tolist()
    return x


def _close(got, want, rtol=2e-6, atol=1e-7):
    for tensor, rows in zip(got, want):
        for a, b in zip(tensor.tolist(), rows):
            for av, bv in zip(a, b):
                assert abs(float(av) - float(bv)) <= atol + rtol * abs(float(bv)), (av, bv)


def _round6(tensors):
    return [[[round(float(v), 6) for v in row] for row in _flat(t)] for t in tensors]


def check_example(solve):
    got = solve(X, EPS, ORIG, MASK, Z, _betas_tensor(), UN, 4, 4, eta=0.5, clip_sample=True)
    want = [
        [[-0.365246, -0.851928], [1.243872, -0.006048], [0.549046, 1.012732]],
        [[0.053841, -0.753922], [1.0, 0.185249], [0.197112, 0.886394]],
        [[0.209584, -0.24202], [1.241198, 0.018253], [0.049418, 0.941732]],
    ]
    assert _round6(got) == want, _round6(got)
    for t in got:
        assert t.dtype == torch.float32 and tuple(t.shape) == (3, 2)


check_example.example = True


def check_hand_spot(solve):
    """mask=1 的格子完全不经过网络：sqrt(0.769120)*(-0.8) + sqrt(0.230880)*0.7 = -0.365246。"""
    got = solve(X, EPS, ORIG, MASK, Z, _betas_tensor(), UN, 4, 4, eta=0.5, clip_sample=True)
    _, _, ab_p, _, _ = _coeffs(BETAS64, 4, 4)
    known = math.sqrt(ab_p) * float(ORIG[0][0]) + math.sqrt(1 - ab_p) * float(Z[0][0])
    assert abs(float(got[0][0][0]) - known) < 1e-6, (float(got[0][0][0]), known)


def check_prev_alpha_is_one_below_first_jump(solve):
    """t < n 时 ᾱ_prev 取 1.0（上游的 final_alpha_cumprod），未知区退化成 x̂₀、已知区退化成原图。

    于是 prev_sample 逐位等于 mask·原图 + (1-mask)·x̂₀——可以直接拿本次返回的第二项核对。
    """
    for t in (0, 1):
        for eta in (0.0, 0.5, 1.0):
            for z in (torch.zeros_like(Z), Z):
                got = solve(X, EPS, ORIG, MASK, z, _betas_tensor(), UN, t, 4, eta=eta, clip_sample=True)
                want = MASK * ORIG + (1.0 - MASK) * got[1]
                assert torch.equal(got[0], want), (t, eta, _round6([got[0], want]))
    # 误用 ᾱ[0]（=0.95）而不是 1.0 会露馅：mask=1 的格子应逐位等于原图
    got = solve(X, EPS, ORIG, MASK, Z, _betas_tensor(), UN, 1, 4, eta=0.5, clip_sample=True)
    assert float(got[0][0][0]) == float(ORIG[0][0]), float(got[0][0][0])
    assert float(got[0][1][1]) == float(ORIG[1][1]), float(got[0][1][1])
    # t=2（第一次跳跃正好落在 ᾱ[0] 上）就不再是 1.0 了，此时 mask=1 的格子必然偏离原图
    got = solve(X, EPS, ORIG, MASK, Z, _betas_tensor(), UN, 2, 4, eta=0.5, clip_sample=True)
    assert float(got[0][0][0]) != float(ORIG[0][0]), float(got[0][0][0])


def check_known_region_ignores_model_and_eta(solve):
    """mask 全 1：输出与 model_output / clip_sample / eta 无关，只由原图、噪声与调度表决定。"""
    ones = torch.ones_like(MASK)
    base = solve(X, EPS, ORIG, ones, Z, _betas_tensor(), UN, 4, 4, eta=0.5, clip_sample=True)
    other_eps = torch.tensor([[-1.0, 2.0], [0.0, 0.5], [0.25, -0.75]], dtype=torch.float32)
    for eta in (0.0, 0.5, 1.0):
        for clip in (True, False):
            alt = solve(X, other_eps, ORIG, ones, Z, _betas_tensor(), UN, 4, 4, eta=eta, clip_sample=clip)
            assert torch.equal(alt[0], base[0]), (eta, clip)
    alt = solve(X, EPS, ORIG + 0.25, ones, Z, _betas_tensor(), UN, 4, 4, eta=0.5, clip_sample=True)
    assert not torch.equal(alt[0], base[0])


def check_unknown_region_ignores_original(solve):
    zeros = torch.zeros_like(MASK)
    base = solve(X, EPS, ORIG, zeros, Z, _betas_tensor(), UN, 4, 4, eta=0.5, clip_sample=True)
    for delta in (0.3, -0.7):
        alt = solve(X, EPS, ORIG + delta, zeros, Z, _betas_tensor(), UN, 4, 4, eta=0.5, clip_sample=True)
        assert torch.equal(alt[0], base[0]), delta


def check_single_noise_feeds_both_terms(solve):
    """同一张 noise 既进已知区也进 η 项：令 x=ε=原图=0 时结果是 noise 的逐点线性组合。"""
    z = torch.tensor([[0.6, -0.9], [0.15, 0.4], [-0.25, 0.8]], dtype=torch.float32)
    zero = torch.zeros_like(z)
    soft = torch.tensor([[0.25, 0.75], [0.5, 0.0], [1.0, 0.6]], dtype=torch.float32)
    for t, n_in, eta in ((4, 4, 0.5), (6, 4, 1.0), (5, 8, 0.75), (0, 4, 1.0), (4, 4, 0.0), (3, 8, 0.9)):
        n, ab_t, ab_p, var, _ = _coeffs(BETAS64, t, n_in)
        if t + n > len(BETAS64):
            continue
        std = eta * math.sqrt(var)
        add = std if t > 0 and eta > 0 else 0.0
        coef = (soft.to(torch.float64) * math.sqrt(1 - ab_p) + (1.0 - soft.to(torch.float64)) * add).to(torch.float32)
        got = solve(zero, zero, zero, soft, z, _betas_tensor(), UN[:n], t, n_in, eta=eta, clip_sample=True)
        assert torch.allclose(got[0], coef * z, rtol=1e-6, atol=1e-7), (t, n_in, eta, got[0].tolist())
        # 把两个系数解耦：mask 全 0 只剩 η 项，mask 全 1 只剩 sqrt(1-ᾱ_prev)
        if add > 0:
            unk = solve(zero, zero, zero, torch.zeros_like(soft), z, _betas_tensor(), UN[:n], t, n_in, eta=eta)[0]
            knw = solve(zero, zero, zero, torch.ones_like(soft), z, _betas_tensor(), UN[:n], t, n_in, eta=eta)[0]
            assert torch.allclose(unk, (add * z), rtol=1e-6, atol=1e-7)
            assert torch.allclose(knw, (math.sqrt(1 - ab_p) * z), rtol=1e-6, atol=1e-7)
            assert not torch.allclose(unk, knw, atol=1e-3)


def check_eta_zero_is_closed_ddim(solve):
    """eta=0：未知区 = sqrt(ᾱ_prev)·x̂₀ + sqrt(1-ᾱ_prev)·ε，与 noise 无关；已知区仍然要加噪。"""
    _, ab_t, ab_p, _, _ = _coeffs(BETAS64, 4, 4)
    est = torch.clamp(
        (X.to(torch.float64) - (1 - ab_t) ** 0.5 * EPS.to(torch.float64)) / ab_t**0.5, -1.0, 1.0
    )
    unknown = (ab_p**0.5 * est + (1 - ab_p) ** 0.5 * EPS.to(torch.float64)).to(torch.float32)
    zeros = torch.zeros_like(MASK)
    a = solve(X, EPS, ORIG, zeros, Z, _betas_tensor(), UN, 4, 4, eta=0.0, clip_sample=True)[0]
    b = solve(X, EPS, ORIG, zeros, -Z, _betas_tensor(), UN, 4, 4, eta=0.0, clip_sample=True)[0]
    assert torch.equal(a, b)
    assert torch.allclose(a, unknown, rtol=1e-6, atol=1e-7), _round6([a, unknown])
    ones = torch.ones_like(MASK)
    c = solve(X, EPS, ORIG, ones, Z, _betas_tensor(), UN, 4, 4, eta=0.0, clip_sample=True)[0]
    known = (ab_p**0.5 * ORIG.to(torch.float64) + (1 - ab_p) ** 0.5 * Z.to(torch.float64)).to(torch.float32)
    assert torch.allclose(c, known, rtol=1e-6, atol=1e-7), _round6([c, known])
    # 混合掩码就是两段逐格相加
    mixed = solve(X, EPS, ORIG, MASK, Z, _betas_tensor(), UN, 4, 4, eta=0.0, clip_sample=True)[0]
    assert torch.equal(mixed, MASK * c + (1.0 - MASK) * a)


def check_direction_uses_prev_alpha_not_t(solve):
    """方向项系数是 sqrt(1 - ᾱ_prev - std²)：误用 ᾱ_t 会得到 0.725941 而不是 0.434410。"""
    n, ab_t, ab_p, var, _ = _coeffs(BETAS64, 4, 4)
    std = 0.5 * math.sqrt(var)
    right = math.sqrt((1 - ab_p) - std**2)
    wrong = math.sqrt((1 - ab_t) - std**2)
    assert abs(right - wrong) > 0.1, (right, wrong)
    zero = torch.zeros_like(X)
    got = solve(zero, EPS, ORIG, torch.zeros_like(MASK), torch.zeros_like(Z), _betas_tensor(), UN, 4, 4,
                eta=0.5, clip_sample=False)[0]
    net = -math.sqrt((1 - ab_t) / ab_t) * math.sqrt(ab_p) + right
    assert torch.allclose(got, (net * EPS).to(torch.float32), rtol=1e-5, atol=1e-6), _round6([got])
    # std 也容易被写成 eta*var（少开根号）：两者在本例差 0.24
    assert abs(std - 0.5 * var) > 0.1, (std, 0.5 * var)


def check_clip_binds_only_where_needed(solve):
    clipped = solve(X, EPS, ORIG, MASK, Z, _betas_tensor(), UN, 4, 4, eta=0.5, clip_sample=True)
    raw = solve(X, EPS, ORIG, MASK, Z, _betas_tensor(), UN, 4, 4, eta=0.5, clip_sample=False)
    assert float(raw[1][1][0]) > 1.0, float(raw[1][1][0])
    assert float(clipped[1][1][0]) == 1.0
    for i in range(3):
        for j in range(2):
            clamped = float(clipped[1][i][j]) != float(raw[1][i][j])
            if float(MASK[i][j]) == 1.0:
                assert float(clipped[0][i][j]) == float(raw[0][i][j]), (i, j)
            elif clamped:
                assert float(clipped[0][i][j]) != float(raw[0][i][j]), (i, j)
            else:
                assert float(clipped[0][i][j]) == float(raw[0][i][j]), (i, j)
    ones = torch.ones_like(MASK)
    assert torch.equal(
        solve(X, EPS, ORIG, ones, Z, _betas_tensor(), UN, 4, 4, eta=0.5, clip_sample=True)[0],
        solve(X, EPS, ORIG, ones, Z, _betas_tensor(), UN, 4, 4, eta=0.5, clip_sample=False)[0],
    )


def check_undo_is_per_substep_not_one_big_jump(solve):
    """回跳是 n 个 β 的子步：累积系数 = sqrt(prod_{i=t..t+n-1}(1-beta_i))，不是 sqrt(ᾱ_{t+n}/ᾱ_t)。"""
    betas = _betas_tensor()
    for t, n_in in ((4, 4), (2, 2), (6, 4), (0, 1), (5, 8), (3, 8)):
        n, _, _, _, _ = _coeffs(BETAS64, t, n_in)
        if t + n > len(BETAS64):
            continue
        zero_un = torch.zeros((n,) + tuple(X.shape), dtype=torch.float32)
        out = solve(X, EPS, ORIG, MASK, Z, betas, zero_un, t, n_in, eta=0.5, clip_sample=True)[2]
        r = 1.0
        for i in range(n):
            r *= 1.0 - BETAS64[t + i]
        assert torch.allclose(out, (math.sqrt(r) * X).to(torch.float32), rtol=1e-5, atol=1e-7), (t, n_in)
        if t + n < len(BETAS64):
            abar = _abar_table(BETAS64)
            off_by_one = math.sqrt(abar[t + n] / abar[t])
            assert abs(off_by_one - math.sqrt(r)) > 1e-4, (t, n_in, off_by_one, math.sqrt(r))


def check_undo_uses_every_draw(solve):
    """n 个子步各用 undo_noise[i]，顺序也不能颠倒。"""
    n, _, _, _, _ = _coeffs(BETAS64, 2, 2)
    un = torch.randn((n, 3, 2), generator=torch.Generator().manual_seed(11))
    base = solve(X, EPS, ORIG, MASK, Z, _betas_tensor(), un, 2, 2, eta=0.0)[2]
    for i in range(n):
        mod = un.clone()
        mod[i] = 0.0
        assert not torch.equal(solve(X, EPS, ORIG, MASK, Z, _betas_tensor(), mod, 2, 2, eta=0.0)[2], base), i
    swapped = un.flip(0).clone()
    assert not torch.equal(solve(X, EPS, ORIG, MASK, Z, _betas_tensor(), swapped, 2, 2, eta=0.0)[2], base)
    # n=1 时只有一张噪声，回跳化成一次的 sqrt(1-beta_t)·x + sqrt(beta_t)·un
    one = solve(X, EPS, ORIG, MASK, Z, _betas_tensor(), un[:1], 2, 8, eta=0.0)[2]
    want = (math.sqrt(1 - BETAS64[2]) * X + math.sqrt(BETAS64[2]) * un[0]).to(torch.float32)
    assert torch.allclose(one, want, rtol=1e-6, atol=1e-7), _round6([one, want])


def check_against_oracle(solve):
    g = torch.Generator().manual_seed(20261030)
    schedules = [
        [0.05, 0.08, 0.12, 0.18, 0.25, 0.33, 0.40, 0.50],
        [0.001, 0.01, 0.05, 0.2, 0.45],
        [0.30, 0.02, 0.10, 0.40, 0.03, 0.15, 0.25, 0.35, 0.05, 0.45, 0.08, 0.20],
        [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.004, 0.008],
        [0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
    ]
    tried = 0
    for betas_list in schedules:
        T = len(betas_list)
        betas = torch.tensor(betas_list, dtype=torch.float32)
        for n_in in [d for d in range(1, T + 1) if T % d == 0]:
            for t in range(0, T - T // n_in + 1):
                _, _, _, _, cap = _coeffs(betas_list, t, n_in)
                etas = [0.0, 0.5, 1.0, 2.3] if math.isinf(cap) else [0.0, 0.5 * cap, 0.99 * cap]
                for eta in etas:
                    for clip in (True, False):
                        for mode in (0, 1):
                            h = 2 + (tried % 3)
                            w = 2 + ((tried // 3) % 3)
                            shape = (h, w)
                            x = torch.randn(shape, generator=g)
                            eps = torch.randn(shape, generator=g)
                            orig = torch.clamp(torch.randn(shape, generator=g), -1.0, 1.0)
                            noise = torch.randn(shape, generator=g)
                            if mode == 0:
                                mask = (torch.rand(shape, generator=g) > 0.5).to(torch.float32)
                            else:
                                mask = torch.rand(shape, generator=g)
                            nn = T // n_in
                            undo = torch.randn((nn,) + shape, generator=g)
                            got = solve(x, eps, orig, mask, noise, betas, undo, t, n_in, eta=eta, clip_sample=clip)
                            want = _oracle(x, eps, orig, mask, noise, betas_list, undo, t, n_in, eta, clip)
                            _close(got, want)
                            tried += 1
    assert tried > 300, tried


def check_dtypes_and_shapes(solve):
    g = torch.Generator().manual_seed(7)
    big = torch.randn((2, 5, 4), generator=g)
    betas = _betas_tensor()
    out = solve(
        big,
        torch.randn(big.shape, generator=g),
        torch.clamp(torch.randn(big.shape, generator=g), -1, 1),
        (torch.rand(big.shape, generator=g) > 0.5).to(torch.float32),
        torch.randn(big.shape, generator=g),
        betas,
        torch.randn((2,) + tuple(big.shape), generator=g),
        6,
        4,
        eta=0.5,
    )
    for t in out:
        assert t.dtype == torch.float32 and tuple(t.shape) == tuple(big.shape)
    assert float(out[1].min()) >= -1.0 and float(out[1].max()) <= 1.0
    # 一维 batch 也合法
    out1 = solve(
        torch.tensor([0.5, -0.25]),
        torch.tensor([0.1, 0.2]),
        torch.tensor([0.3, -0.4]),
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.7, -0.6]),
        betas,
        torch.tensor([[0.1, 0.2], [-0.3, 0.4]]),
        4,
        4,
    )
    assert tuple(out1[0].shape) == (2,)


def _poison(t, value):
    out = t.clone()
    out[0][0] = value
    return out


def _poison_1d(t, value):
    out = t.clone()
    out[0] = value
    return out


def check_invalid_inputs(solve):
    b = _betas_tensor()
    bad = [
        ("sample 非张量", lambda: solve([[0.1, 0.2]], EPS[:1], ORIG[:1], MASK[:1], Z[:1], b, UN[:1], 0, 4)),
        ("sample 0 维", lambda: solve(torch.tensor(1.0), None, None, None, None, b, None, 0, 4)),
        ("model_output 形状不一致", lambda: solve(X, EPS[:2], ORIG, MASK, Z, b, UN, 4, 4)),
        ("original_image 形状不一致", lambda: solve(X, EPS, ORIG[:2], MASK, Z, b, UN, 4, 4)),
        ("mask 形状不一致", lambda: solve(X, EPS, ORIG, MASK[:2], Z, b, UN, 4, 4)),
        ("noise 形状不一致", lambda: solve(X, EPS, ORIG, MASK, Z[:2], b, UN, 4, 4)),
        ("mask 大于 1", lambda: solve(X, EPS, ORIG, _poison(MASK, 2.0), Z, b, UN, 4, 4)),
        ("mask 为负", lambda: solve(X, EPS, ORIG, _poison(MASK, -0.1), Z, b, UN, 4, 4)),
        ("sample 含 NaN", lambda: solve(_poison(X, float("nan")), EPS, ORIG, MASK, Z, b, UN, 4, 4)),
        ("noise 含 inf", lambda: solve(X, EPS, ORIG, MASK, _poison(Z, float("inf")), b, UN, 4, 4)),
        ("betas 非张量", lambda: solve(X, EPS, ORIG, MASK, Z, BETAS64, UN, 4, 4)),
        ("betas 二维", lambda: solve(X, EPS, ORIG, MASK, Z, b.view(4, 2), UN, 4, 4)),
        ("betas 含负", lambda: solve(X, EPS, ORIG, MASK, Z, _poison_1d(b, -0.01), UN, 4, 4)),
        ("betas 含 1", lambda: solve(X, EPS, ORIG, MASK, Z, _poison_1d(b, 1.0), UN, 4, 4)),
        ("betas 含 NaN", lambda: solve(X, EPS, ORIG, MASK, Z, _poison_1d(b, float("nan")), UN, 4, 4)),
        ("betas 使 abar 触 0", lambda: solve(X, EPS, ORIG, MASK, Z, torch.full((1200,), 0.9), UN, 0, 1200)),
        ("undo_noise 非张量", lambda: solve(X, EPS, ORIG, MASK, Z, b, [[[0.0, 0.0]]], 4, 4)),
        ("undo_noise 少一张", lambda: solve(X, EPS, ORIG, MASK, Z, b, UN[:1], 4, 4)),
        ("undo_noise 多一张", lambda: solve(X, EPS, ORIG, MASK, Z, b, torch.cat([UN, UN[:1]]), 4, 4)),
        ("undo_noise 空间形状错", lambda: solve(X, EPS, ORIG, MASK, Z, b, torch.zeros((2, 3, 5)), 4, 4)),
        ("timestep 为浮点", lambda: solve(X, EPS, ORIG, MASK, Z, b, UN, 4.0, 4)),
        ("timestep 为 bool", lambda: solve(X, EPS, ORIG, MASK, Z, b, UN, True, 4)),
        ("timestep 为负", lambda: solve(X, EPS, ORIG, MASK, Z, b, UN, -1, 4)),
        ("timestep + n 越界", lambda: solve(X, EPS, ORIG, MASK, Z, b, UN, 7, 4)),
        ("timestep 越界", lambda: solve(X, EPS, ORIG, MASK, Z, b, UN, 8, 4)),
        ("num_inference_steps 不整除", lambda: solve(X, EPS, ORIG, MASK, Z, b, torch.zeros((3, 3, 2)), 0, 3)),
        ("num_inference_steps 为 0", lambda: solve(X, EPS, ORIG, MASK, Z, b, UN, 0, 0)),
        ("num_inference_steps 为浮点", lambda: solve(X, EPS, ORIG, MASK, Z, b, UN, 4, 4.0)),
        ("num_inference_steps 为 bool", lambda: solve(X, EPS, ORIG, MASK, Z, b, UN, 4, True)),
        ("eta 为负", lambda: solve(X, EPS, ORIG, MASK, Z, b, UN, 4, 4, eta=-0.5)),
        ("eta 为 NaN", lambda: solve(X, EPS, ORIG, MASK, Z, b, UN, 4, 4, eta=float("nan"))),
        ("eta 为 bool", lambda: solve(X, EPS, ORIG, MASK, Z, b, UN, 4, 4, eta=True)),
        ("eta 为字符串", lambda: solve(X, EPS, ORIG, MASK, Z, b, UN, 4, 4, eta="0.5")),
        ("eta 过大", lambda: solve(X, EPS, ORIG, MASK, Z, b, UN, 4, 4, eta=3.0)),
        ("clip_sample 非 bool", lambda: solve(X, EPS, ORIG, MASK, Z, b, UN, 4, 4, clip_sample=1)),
    ]
    for name, fn in bad:
        try:
            fn()
        except ValueError:
            continue
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(f"{name}: 抛了 {type(exc).__name__}，应为 ValueError") from exc
        raise AssertionError(f"{name}: 未抛 ValueError")
