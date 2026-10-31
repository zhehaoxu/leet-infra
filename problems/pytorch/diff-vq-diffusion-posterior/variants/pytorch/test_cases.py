import math

import torch


LOG_LEAK = -69.07755278982137  # log(1e-30)
E_INV = math.exp(-1.0)  # 该处 Gumbel 噪声恰为 0

C3 = 3
AT = [0.80, 0.60, 0.40, 0.20]
CT = [0.10, 0.20, 0.30, 0.40]
ATT = [0.80, 0.48, 0.192, 0.0384, 1.0]  # 末位是永远读不到的哨兵
CTT = [0.10, 0.28, 0.496, 0.6976, 0.0]

MO = [[[-0.2, 0.3, -1.0, 0.5], [0.4, -0.7, 0.1, -0.3]]]
SAMPLE = [[0, 2, 1, 2]]


def _t(nested, dtype=torch.float64):
    return torch.tensor(nested, dtype=dtype)


def _flat(x):
    return x.tolist() if isinstance(x, torch.Tensor) else x


def _ints(x):
    out = []
    stack = [_flat(x)]
    while stack:
        v = stack.pop()
        if isinstance(v, list):
            stack.extend(v)
        else:
            out.append(int(v))
    return out


def _rows(logits):
    return [[[round(float(v), 6) for v in row] for row in _flat(t)] for t in logits]


def _close(a, b, rtol=1e-9, atol=1e-12):
    a, b = _flat(a), _flat(b)
    if isinstance(a, list):
        assert len(a) == len(b), (len(a), len(b))
        for x, y in zip(a, b):
            _close(x, y, rtol, atol)
        return
    av, bv = float(a), float(b)
    assert abs(av - bv) <= atol + rtol * abs(bv), (av, bv)


def _call(solve, mo=MO, sample=SAMPLE, t=1, num_classes=C3, at=AT, ct=CT, att=ATT, ctt=CTT, uniform=None):
    if uniform is None:
        n_img, length = len(mo), len(mo[0][0])
        n_row = num_classes - (1 if t == 0 else 0)
        uniform = [[[E_INV] * length] * n_row] * n_img
    return solve(
        _t(mo), _t(sample, torch.int64), t,
        _t(at), _t(ct), _t(att), _t(ctt), num_classes, _t(uniform),
    )


def oracle(mo, sample, t, at, ct, att, ctt, num_classes):
    """独立 oracle：概率空间、显式转移矩阵，不走 log 空间的 logaddexp 技巧。"""
    k = num_classes - 1
    if t == 0:
        return [[list(col) for col in batch] for batch in mo]
    out = []
    for p_log, obs_row in zip(mo, sample):
        cols = [[] for _ in range(num_classes)]
        for l, obs in enumerate(obs_row):
            p = [math.exp(p_log[j][l]) for j in range(k)]
            A, gam = att[t], ctt[t]
            B = (1.0 - A - gam) / k
            r = [p[i] / (gam if obs == k else (A + B if i == obs else B)) for i in range(k)]
            s = sum(r)
            w = [x / s for x in r]
            a, c = at[t], ct[t]
            b = (1.0 - a - c) / k
            A2, C2 = att[t - 1], ctt[t - 1]
            B2 = (1.0 - A2 - C2) / k
            for j in range(k):
                mix = B2 + A2 * w[j]  # 掩码是吸收态：只有"上一刻就是它"的那一列带 A2
                m = c if obs == k else (a + b if j == obs else b)
                cols[j].append(m * mix * s)
            cols[k].append((1.0 if obs == k else 1e-30) * C2 * s)
        out.append([[math.log(v) for v in col] for col in cols])
    return out


def _expected_prev(logits, uniform, noise=None):
    """按参考口径复算逐像素 argmax：平票取最小下标。"""
    logits, uniform = _flat(logits), _flat(uniform)
    if noise is None:
        noise = [[[-math.log(-math.log(u + 1e-30) + 1e-30) for u in row] for row in img] for img in uniform]
    out = []
    for img_l, img_u, img_n in zip(logits, uniform, noise):
        cols = list(zip(*img_l))  # [pixel][class]
        out.append([max(range(len(row)), key=lambda j: (row[j] + img_n[j][l], -j))
                    for l, row in enumerate(cols)])
    return out


def check_example(solve):
    prev, logits = _call(solve, t=0)
    assert _rows(logits) == [[[-0.2, 0.3, -1.0, 0.5], [0.4, -0.7, 0.1, -0.3]]], _rows(logits)
    assert _flat(prev) == [[1, 0, 1, 0]], _flat(prev)

    prev, logits = _call(solve, t=1)
    assert _rows(logits) == [
        [[0.22076, -0.177582, -1.310108, 0.027097],
         [0.061595, -1.050658, 0.185034, -0.675965],
         [-68.755728, -0.416358, -69.78935, -0.158519]]], _rows(logits)
    assert _flat(prev) == [[0, 0, 1, 0]], _flat(prev)

    prev, logits = _call(solve, t=2)
    assert _rows(logits) == [
        [[0.341843, -0.642624, -1.31155, -0.427565],
         [-0.102015, -1.279465, 0.185357, -0.945435],
         [-67.872666, 0.041475, -68.639611, 0.299314]]], _rows(logits)
    assert _flat(prev) == [[0, 2, 1, 2]], _flat(prev)

    prev, logits = _call(solve, t=3)
    assert _rows(logits) == [
        [[0.317762, -1.159089, -0.925513, -0.928227],
         [-0.065598, -1.514885, 0.073922, -1.219759],
         [-66.999512, 0.272192, -67.551656, 0.530031]]], _rows(logits)
    assert _flat(prev) == [[0, 2, 1, 2]], _flat(prev)


check_example.example = True


def check_example_with_gumbel_noise(solve):
    u3 = [[[0.50, 0.10, 0.90, 0.20], [0.30, 0.70, 0.40, 0.60], [0.80, 0.05, 0.25, 0.65]]]
    u2 = [[[0.50, 0.10, 0.90, 0.20], [0.30, 0.70, 0.40, 0.60]]]
    prev0, _ = _call(solve, t=0, uniform=u2)
    assert _flat(prev0) == [[1, 1, 0, 1]], _flat(prev0)
    for t in (1, 2, 3):
        prev, logits = _call(solve, t=t, uniform=u3)
        assert _flat(prev) == [[0, 1, 0, 2]], _flat(prev)
        _, clean = _call(solve, t=t)
        _close(logits, clean)  # 噪声只改采样结果，不改返回的分布


def check_t_zero_is_verbatim_model_output(solve):
    """t=0 是捷径：分布原样返回，掩码类压根不在候选里，少一维。"""
    for num_classes in (3, 5):
        k = num_classes - 1
        mo = [[[i * 0.9 - 1.4 + j * 0.21 for i in range(3)] for j in range(k)]]
        prev, logits = _call(solve, mo=mo, sample=[[0, 1, 2]], t=0, num_classes=num_classes)
        assert tuple(logits.shape) == (1, k, 3), tuple(logits.shape)
        assert _flat(logits) == mo, "t=0 必须逐字返回 model_output"
        assert max(_ints(prev)) <= k - 1, "t=0 不可能采出掩码类"


def check_column_mass_is_preserved(solve):
    """每个像素 exp 后按类别求和 = 输入 model_output 的 exp 之和：中间那步归一化是齐次的。"""
    mass_in = [sum(math.exp(v) for v in col) for col in zip(*MO[0])]
    for t in (0, 1, 2, 3):
        _, logits = _call(solve, t=t)
        rows = _flat(logits)[0]
        for l in range(4):
            mass_out = sum(math.exp(rows[j][l]) for j in range(len(rows)))
            assert abs(mass_out - mass_in[l]) <= 1e-9 * mass_in[l], (t, l, mass_out, mass_in[l])
    z = [math.log(math.exp(MO[0][0][l]) + math.exp(MO[0][1][l])) for l in range(4)]
    normed = [[[MO[0][j][l] - z[l] for l in range(4)] for j in range(2)]]
    for t in (0, 1, 2, 3):
        _, logits = _call(solve, mo=normed, t=t)
        rows = _flat(logits)[0]
        for l in range(4):
            total = sum(math.exp(rows[j][l]) for j in range(len(rows)))
            assert abs(total - 1.0) < 1e-12, (t, l, total)


def check_log_shift_is_equivariant(solve):
    """一像素所有类同加常数，输出整体平移同一常数。"""
    for t in (1, 2, 3):
        _, base = _call(solve, t=t)
        shifted = [[[v + 1.75 for v in row] for row in MO[0]]]
        _, got = _call(solve, mo=shifted, t=t)
        for j in range(C3):
            _close(got[0, j], [v + 1.75 for v in _flat(base[0, j])])


def check_mask_row_is_clamped_not_negative_infinity(solve):
    """q(x_t=已知 | x_{t-1}=掩码) 真值是 0，实现里是钳制后的 log(1e-30) ≈ -69.0776。"""
    for t in (1, 2, 3):
        _, logits = _call(solve, t=t)
        row = _flat(logits[0, C3 - 1])
        assert all(math.isfinite(v) for v in row), row
        # 观测为掩码的像素（下标 1、3）掩码行是正常量级，其余两格带钳制项
        assert row[0] < -60 and row[2] < -60, row
        assert -5 < row[1] < 5 and -5 < row[3] < 5, row
        assert abs(row[0] - LOG_LEAK) < 5.0, row
    _, logits = _call(solve, t=1)
    assert abs(logits[0, C3 - 1, 0].item() + 68.755728) < 1e-5


def check_known_pixel_never_becomes_masked(solve):
    """未掩码像素往掩码类只有 1e-30 量级，噪声顶格也翻不过来。"""
    u_hi = 1.0 - 1e-15  # 仍可表示且 <1，Gumbel 噪声约 +34.5
    cases = [
        (3, [0.8, 0.6, 0.4], [0.1, 0.2, 0.3], [0.8, 0.48, 0.192, 1.0], [0.1, 0.28, 0.496, 0.0]),
        (4, [0.9, 0.7, 0.5], [0.05, 0.15, 0.25], [0.9, 0.63, 0.315, 1.0], [0.05, 0.1925, 0.389, 0.0]),
        (2, [0.6, 0.4], [0.2, 0.3], [0.6, 0.24, 1.0], [0.2, 0.44, 0.0]),
    ]
    for num_classes, at, ct, att, ctt in cases:
        k = num_classes - 1
        for t in range(1, len(at)):
            mo = [[[0.0] * k for _ in range(k)]]
            sample = [[i % k for i in range(k)]]  # 全部落在已知类
            uniform = [[[0.5] * k for _ in range(k)] + [[u_hi] * k]]
            prev, _ = _call(solve, mo=mo, sample=sample, t=t, num_classes=num_classes,
                            at=at, ct=ct, att=att, ctt=ctt, uniform=uniform)
            assert all(v != k for v in _ints(prev)), (num_classes, t, _ints(prev))


def check_masked_pixel_stays_masked_when_cumulative_gamma_is_large(solve):
    """γ̄_{t-1} 大时，掩码像素的掩码行就是分布里的最大值：吸收态把质量留在掩码上。"""
    at = [0.20, 0.15, 0.10, 0.08]
    ct = [0.70, 0.75, 0.80, 0.85]
    att = [0.20, 0.15, 0.10, 0.05, 1.0]
    ctt = [0.70, 0.78, 0.85, 0.90, 0.0]
    mo = [[[0.0, 0.0], [0.0, 0.0]]]
    for t in (1, 2, 3):
        _, logits = _call(solve, mo=mo, sample=[[2, 0]], t=t, at=at, ct=ct, att=att, ctt=ctt)
        rows = _flat(logits)[0]
        best_masked = max(range(C3), key=lambda j: rows[j][0])
        best_known = max(range(C3), key=lambda j: rows[j][1])
        assert best_masked == 2, (t, rows)
        assert best_known != 2, (t, rows)
        assert rows[2][0] > rows[2][1] + 60, (t, rows)


def check_hand_spot_t1(solve):
    """t=1 第一格纯手算：钉住累积下标 t、回推下标 t-1、b=(1-a-c)/k 与钳制项。"""
    p = [math.exp(-0.2), math.exp(0.4)]
    A, gam = ATT[1], CTT[1]
    a, c = AT[1], CT[1]
    A2, C2 = ATT[0], CTT[0]
    B, b, B2 = (1 - A - gam) / 2, (1 - a - c) / 2, (1 - A2 - C2) / 2
    r = [p[0] / (A + B), p[1] / B]  # 观测 x_t = 0
    s = sum(r)
    w = [x / s for x in r]
    out0 = (a + b) * (B2 + A2 * w[0]) * s
    out1 = b * (B2 + A2 * w[1]) * s
    outm = 1e-30 * C2 * s
    _, logits = _call(solve, t=1)
    assert abs(math.log(out0) - logits[0, 0, 0].item()) < 1e-9, math.log(out0)
    assert abs(math.log(out1) - logits[0, 1, 0].item()) < 1e-9, math.log(out1)
    assert abs(math.log(outm) - logits[0, 2, 0].item()) < 1e-9, math.log(outm)
    # b 的分母是已知类个数 C-1，不是 C 也不是 1
    assert abs(b - 0.1) < 1e-12 and abs(B - 0.12) < 1e-12 and abs(B2 - 0.05) < 1e-12, (b, B, B2)


def check_gumbel_at_u_zero_uses_the_inner_clamp(solve):
    """u=0 时噪声是 -log(69.0776) = -4.2349，不是 -inf：少了内层 +1e-30 就会翻车。"""
    noise_at = lambda u: -math.log(-math.log(u + 1e-30) + 1e-30)  # noqa: E731
    assert abs(noise_at(0.0) + 4.235229826910111) < 1e-9, noise_at(0.0)
    assert abs(noise_at(E_INV)) < 1e-12, noise_at(E_INV)
    assert noise_at(1.0 - 1e-15) > 34.0, noise_at(1.0 - 1e-15)
    mo = [[[10.0], [-4.5]]]
    uniform = [[[0.0], [1e-18]]]
    prev, _ = _call(solve, mo=mo, sample=[[0]], t=0, uniform=uniform)
    # 钳制下：10-4.2349=5.765 > -4.5-3.7237=-8.224 → 类 0；按 -log(-log(u)) 直算则类 0 得 -inf → 类 1
    assert _flat(prev) == [[0]], _flat(prev)


def check_argmax_tie_takes_first_index(solve):
    """完全打平时 torch.argmax 取下标最小者。"""
    n = 5
    mo = [[[0.0] * n, [0.0] * n]]
    prev, logits = _call(solve, mo=mo, sample=[[1] * n], t=0)
    assert _flat(logits) == mo
    assert _flat(prev) == [[0] * n], _flat(prev)
    prev, _ = _call(solve, mo=mo, sample=[[1] * n], t=0, uniform=[[[0.2] * n, [0.2] * n]])
    assert _flat(prev) == [[0] * n], _flat(prev)


def check_batch_and_pixel_are_independent(solve):
    """单像素单独解与整批里那一列逐位相同：标量 a/b/c 与掩码列 expand 的广播方向没写反。"""
    for t in (1, 2, 3):
        _, whole = _call(solve, t=t)
        for l in range(4):
            mo_col = [[[MO[0][j][l]] for j in range(C3 - 1)]]
            _, single = _call(solve, mo=mo_col, sample=[[SAMPLE[0][l]]], t=t)
            for j in range(C3):
                _close(single[0, j], [whole[0, j, l].item()])
        mo_stack = MO + MO
        sample_stack = [SAMPLE[0], SAMPLE[0]]
        u = [[[E_INV] * 4] * C3, [[E_INV] * 4] * C3]
        _, stacked = _call(solve, mo=mo_stack, sample=sample_stack, t=t, uniform=u)
        for j in range(C3):
            _close(stacked[1, j], _flat(whole[0, j]))
        assert _flat(stacked[0]) != [] and tuple(stacked.shape) == (2, C3, 4)


def check_prev_sample_matches_noised_argmax(solve):
    """prev_sample 必须等于"分布 + 噪声"的逐像素 argmax，两处都不能少加噪声。"""
    uniform = [[[0.05, 0.6, 0.2, 0.95], [0.4, 0.02, 0.75, 0.1], [0.6, 0.85, 0.3, 0.5]]]
    for t in (0, 1, 2, 3):
        u = [[uniform[0][0], uniform[0][1]]] if t == 0 else uniform
        prev, logits = _call(solve, t=t, uniform=u)
        want = _expected_prev(logits, u)
        assert _flat(prev) == want, (t, _flat(prev), want)


def check_against_oracle(solve):
    schedules = [
        (AT, CT, ATT, CTT, 3),
        ([0.9, 0.7, 0.5, 0.3], [0.02, 0.08, 0.2, 0.35],
         [0.9, 0.63, 0.315, 0.0945, 1.0], [0.02, 0.0964, 0.265, 0.5188, 0.0], 4),
        ([0.5, 0.5], [0.1, 0.1], [0.5, 0.25, 1.0], [0.1, 0.19, 0.0], 2),
        ([0.99, 0.01, 0.4, 0.2, 0.3], [0.001, 0.5, 0.05, 0.3, 0.1],
         [0.99, 0.0099, 0.00396, 0.000792, 0.000238, 1.0],
         [0.001, 0.5005, 0.525475, 0.667833, 0.701046, 0.0], 6),
    ]
    tried = 0
    for at, ct, att, ctt, num_classes in schedules:
        T, k = len(at), num_classes - 1
        for t in range(T):
            for b in range(2):
                L = 1 + ((t + b + num_classes) % 4)
                n_row = k if t == 0 else num_classes
                mo, sample = [], []
                for _ in range(b + 1):
                    mo.append([[(i * 0.7 - 1.3 + j * 0.31 + t * 0.11) for i in range(L)] for j in range(k)])
                    sample.append([(i * 3 + b) % num_classes for i in range(L)])
                uniform = [[[((i + 1) * 0.137 + (j + 1) * 0.053 + b * 0.021) % 1.0 for i in range(L)]
                            for j in range(n_row)] for _ in range(b + 1)]
                prev, logits = _call(solve, mo=mo, sample=sample, t=t, num_classes=num_classes,
                                     at=at, ct=ct, att=att, ctt=ctt, uniform=uniform)
                _close(logits, oracle(mo, sample, t, at, ct, att, ctt, num_classes))
                assert tuple(logits.shape) == (b + 1, n_row, L), tuple(logits.shape)
                assert _flat(prev) == _expected_prev(logits, uniform), (t, b)
                tried += 1
    assert tried >= 30, tried


def check_dtypes_and_shapes(solve):
    at = [0.7, 0.6, 0.5, 0.4, 0.3]
    ct = [0.05, 0.08, 0.10, 0.12, 0.15]
    att = [0.7, 0.42, 0.21, 0.084, 0.0252, 1.0]
    ctt = [0.05, 0.12, 0.2, 0.28, 0.35, 0.0]
    mo = torch.randn((2, 5, 3), dtype=torch.float32)
    sample = torch.randint(0, 6, (2, 3), dtype=torch.int64)
    u = torch.rand((2, 6, 3), dtype=torch.float32)
    prev, logits = solve(mo, sample, 3, _t(at), _t(ct), _t(att), _t(ctt), 6, u)
    assert prev.dtype == torch.int64 and tuple(prev.shape) == (2, 3), (prev.dtype, tuple(prev.shape))
    assert logits.dtype == torch.float64 and tuple(logits.shape) == (2, 6, 3)
    assert torch.isfinite(logits).all()
    assert prev.min().item() >= 0 and prev.max().item() <= 5
    u0 = torch.rand((2, 5, 3), dtype=torch.float64)
    prev0, logits0 = solve(mo, sample, 0, _t(at), _t(ct), _t(att), _t(ctt), 6, u0)
    assert tuple(logits0.shape) == (2, 5, 3)
    assert prev0.max().item() < 5
    assert logits0.tolist() == mo.double().tolist()
    # int32 / uint8 的 sample 也接受，内部按 int64 处理
    prev1, _ = solve(mo, sample.to(torch.int32), 2, _t(at), _t(ct), _t(att), _t(ctt), 6, u)
    prev2, _ = solve(mo, sample.to(torch.int64), 2, _t(at), _t(ct), _t(att), _t(ctt), 6, u)
    assert prev1.tolist() == prev2.tolist()


def _poison(t, value, index=(0, 0, 0)):
    out = t.clone()
    out[index] = value
    return out


def check_invalid_inputs(solve):
    at, ct, att, ctt = _t(AT), _t(CT), _t(ATT), _t(CTT)
    mo, sample = _t(MO), _t(SAMPLE, torch.int64)
    u3 = _t([[[E_INV] * 4] * 3])
    u2 = _t([[[E_INV] * 4] * 2])

    def run(**kw):
        args = dict(mo=mo, sample=sample, t=1, at=at, ct=ct, att=att, ctt=ctt, nc=C3, u=u3)
        args.update(kw)
        solve(args["mo"], args["sample"], args["t"], args["at"], args["ct"],
              args["att"], args["ctt"], args["nc"], args["u"])

    bad = [
        ("model_output 非张量", lambda: run(mo=MO)),
        ("model_output 二维", lambda: run(mo=_t([[-0.2, 0.3], [0.4, -0.7]]))),
        ("model_output 第 1 维带上了掩码类", lambda: run(mo=_t([[r, [0.1] * 4] for r in MO[0]]))),
        ("model_output 少一类", lambda: run(mo=_t([MO[0][:-1]]), nc=4)),
        ("model_output 含 NaN", lambda: run(mo=_poison(mo, float("nan")))),
        ("model_output 含 inf", lambda: run(mo=_poison(mo, float("inf")))),
        ("model_output 整数 dtype", lambda: run(mo=_t(MO, torch.int64))),
        ("sample 非张量", lambda: run(sample=SAMPLE)),
        ("sample 浮点", lambda: run(sample=_t(SAMPLE))),
        ("sample 形状不符", lambda: run(sample=_t([[0, 2, 1]], torch.int64))),
        ("sample 含负", lambda: run(sample=_poison(sample, -1, (0, 0)))),
        ("sample 掩码类越界", lambda: run(sample=_poison(sample, 3, (0, 0)))),
        ("sample 多一个 batch", lambda: run(sample=_t([[0, 2, 1, 2], [1, 1, 1, 1]], torch.int64))),
        ("sample 是 bool", lambda: run(sample=_t([[0, 2, 1, 2]], torch.int64).bool())),
        ("timestep 为浮点", lambda: run(t=1.0)),
        ("timestep 为 bool", lambda: run(t=True)),
        ("timestep 为负", lambda: run(t=-1)),
        ("timestep 越界", lambda: run(t=4)),
        ("timestep 为字符串", lambda: run(t="1")),
        ("at 非张量", lambda: run(at=AT)),
        ("at 二维", lambda: run(at=at.view(2, 2))),
        ("at 与 ct 长度不一致", lambda: run(ct=_t([0.1, 0.2, 0.3]))),
        ("at 长度不足 2", lambda: run(at=_t([0.6]), ct=_t([0.2]), att=_t([0.6, 1.0]), ctt=_t([0.2, 0.0]), t=0)),
        ("att 少了哨兵", lambda: run(att=_t(ATT[:4]))),
        ("ctt 长度不是 T+1", lambda: run(ctt=_t(CTT * 2))),
        ("at 含 0", lambda: run(at=_poison(at, 0.0, (2,)))),
        ("at 含 1", lambda: run(at=_poison(at, 1.0, (2,)))),
        ("ct 含 1", lambda: run(ct=_poison(ct, 1.0, (0,)))),
        ("ct 含负", lambda: run(ct=_poison(ct, -0.1, (0,)))),
        ("at + ct 不小于 1", lambda: run(ct=_poison(ct, 0.5, (1,)))),
        ("att 读得到区间含 0", lambda: run(att=_poison(att, 0.0, (1,)))),
        ("att 读得到区间含 1", lambda: run(att=_poison(att, 1.0, (2,)))),
        ("ctt 读得到区间含 1", lambda: run(ctt=_poison(ctt, 1.0, (0,)))),
        ("att + ctt 不小于 1", lambda: run(att=_poison(att, 0.75, (1,)))),
        ("at 含 NaN", lambda: run(at=_poison(at, float("nan"), (1,)))),
        ("ctt 含 inf", lambda: run(ctt=_poison(ctt, float("inf"), (2,)))),
        ("num_vec_classes 为 1", lambda: run(nc=1)),
        ("num_vec_classes 为 bool", lambda: run(nc=True)),
        ("num_vec_classes 为浮点", lambda: run(nc=3.0)),
        ("num_vec_classes 与 model_output 不符", lambda: run(nc=5)),
        ("t=0 却给了 C 类噪声", lambda: run(t=0, u=u3)),
        ("t>0 却给了 C-1 类噪声", lambda: run(t=1, u=u2)),
        ("噪声形状少一像素", lambda: run(u=_t([[[E_INV] * 3] * 3]))),
        ("噪声含 1.0", lambda: run(u=_poison(u3, 1.0))),
        ("噪声含负", lambda: run(u=_poison(u3, -0.1))),
        ("噪声非张量", lambda: run(u=[[[0.1] * 4] * 3])),
        ("噪声整数 dtype", lambda: run(u=_t([[[0] * 4] * 3], torch.int64))),
    ]
    for name, fn in bad:
        try:
            fn()
        except ValueError:
            continue
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(f"{name}: 抛了 {type(exc).__name__}，应为 ValueError") from exc
        raise AssertionError(f"{name}: 未抛 ValueError")

    # 合法边界：t=0 用 (B, C-1, L) 的噪声；哨兵位不参与校验
    run(t=0, u=u2)
    run(mo=_t([[[-1e6, 0.0, 1.0, 2.0], [3.0, -4.0, 5.0, -6.0]]]))
