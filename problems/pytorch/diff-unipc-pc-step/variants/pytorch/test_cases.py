import math

import torch

SIG = torch.tensor([8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.125], dtype=torch.float32)
H3 = (
    torch.tensor([[0.4, -0.2], [1.1, 0.3]], dtype=torch.float32),
    torch.tensor([[-0.5, 0.7], [0.2, 0.9]], dtype=torch.float32),
    torch.tensor([[0.6, 0.1], [-0.3, 0.5]], dtype=torch.float32),
)
MT = torch.tensor([[0.2, -0.4], [0.8, 0.15]], dtype=torch.float32)
XL = torch.tensor([[1.3, -0.9], [0.4, 1.7]], dtype=torch.float32)
XT = torch.tensor([[1.1, -0.7], [0.6, 1.5]], dtype=torch.float32)


def _flat(t):
    return [float(v) for v in t.reshape(-1).tolist()]


def _pair(sig, flow):
    if flow:
        return 1.0 - sig, sig
    alpha = 1.0 / math.sqrt(sig * sig + 1.0)
    return alpha, sig * alpha


def _lam(sig, flow):
    a, s = _pair(sig, flow)
    return math.log(a) - math.log(s)


def _eliminate(rows, rhs):
    """高斯消元解 Rhos = b（参考实现用 torch.linalg.solve，这里是独立实现）。"""
    n = len(rhs)
    assert len(rows) == n and all(len(row) == n for row in rows), (len(rows), n)
    mat = [row[:] + [rhs[i]] for i, row in enumerate(rows)]
    for col in range(n):
        piv = col
        for r in range(col + 1, n):
            if abs(mat[r][col]) > abs(mat[piv][col]):
                piv = r
        mat[col], mat[piv] = mat[piv], mat[col]
        pv = mat[col][col]
        for j in range(col, n + 1):
            mat[col][j] /= pv
        for r in range(n):
            if r == col:
                continue
            factor = mat[r][col]
            for j in range(col, n + 1):
                mat[r][j] -= factor * mat[col][j]
    return [mat[r][n] for r in range(n)]


def _basis(rks, order, hh, b_h):
    phi_1 = math.expm1(hh)
    phi_k = phi_1 / hh - 1.0
    fact = 1.0
    rows = []
    b = []
    for i in range(1, order + 1):
        rows.append([rk ** (i - 1) for rk in rks])
        b.append(phi_k * fact / b_h)
        fact *= i + 1
        phi_k = phi_k / hh - 1.0 / fact
    return rows, b, phi_1


def _one_side(sig, step, offset, order, px0, bh, flow, m0, older, x):
    """older[k] 是比 s0 再老 k+1 步的输出；返回该侧结果。"""
    a_s0, s_s0 = _pair(sig[step - offset], flow)
    a_t, s_t = _pair(sig[step + 1 - offset], flow)
    lam_s0 = math.log(a_s0) - math.log(s_s0)
    h = (math.log(a_t) - math.log(s_t)) - lam_s0
    hh = -h if px0 else h
    b_h = hh if bh == "bh1" else math.expm1(hh)
    rks = []
    d1 = []
    for i in range(1, order):
        si = step - (i + offset)
        rk = (_lam(sig[si], flow) - lam_s0) / h
        rks.append(rk)
        d1.append([(v - w) / rk for v, w in zip(older[i - 1], m0)])
    rks.append(1.0)
    rows, b, phi_1 = _basis(rks, order, hh, b_h)
    if order == 1 and offset == 1:
        rhos = [0.5]
    elif order == 2 and offset == 0:
        rhos = [0.5]
    elif order == 1:
        rhos = []
    elif offset == 0:
        rhos = _eliminate([row[:-1] for row in rows[:-1]], b[:-1])
    else:
        rhos = _eliminate(rows, b)
    acc = [0.0] * len(m0)
    for rho, di in zip(rhos, d1):
        acc = [v + rho * d for v, d in zip(acc, di)]
    if offset == 1:
        d1_t = [t - m for t, m in zip(x[1], m0)]
        acc = [v + rhos[-1] * d for v, d in zip(acc, d1_t)]
    if px0:
        return [(s_t / s_s0) * xv - a_t * phi_1 * mv - a_t * b_h * av for xv, mv, av in zip(x[0], m0, acc)]
    return [(a_t / a_s0) * xv - s_t * phi_1 * mv - s_t * b_h * av for xv, mv, av in zip(x[0], m0, acc)]


def _oracle(m_hist, m_t, x_last, x_this, sigmas, step, oc, op, px0, bh, flow, shape):
    hist = [_flat(t) for t in m_hist]
    mt = _flat(m_t)
    xt = _flat(x_this)
    sig = [float(v) for v in sigmas.reshape(-1).tolist()]
    if x_last is None:
        corrected = xt
    else:
        m0 = hist[-1]
        older = [hist[-(i + 1)] for i in range(1, oc)]
        corrected = _one_side(sig, step, 1, oc, px0, bh, flow, m0, older, (_flat(x_last), mt))
    older_p = [hist[-i] for i in range(1, op)]
    prev = _one_side(sig, step, 0, op, px0, bh, flow, mt, older_p, (corrected, None))
    to_t = lambda rows: torch.tensor(rows, dtype=torch.float64).reshape(shape)
    return to_t(corrected), to_t(prev)


def check_example(solve):
    c, p = solve(H3, MT, XL, XT, SIG, 3, 2, 2)
    assert torch.allclose(c, torch.tensor([[1.204902, -0.777599], [0.380135, 1.450320]], dtype=torch.float32),
                           atol=1e-5), c
    assert torch.allclose(p, torch.tensor([[0.762047, -0.782485], [0.844157, 0.906083]], dtype=torch.float32),
                           atol=1e-5), p
    assert c.dtype == torch.float32 and p.dtype == torch.float32
    assert tuple(c.shape) == (2, 2) and tuple(p.shape) == (2, 2)


check_example.example = True


def _close(a, b, atol=2e-5):
    return torch.allclose(a.to(torch.float64), b.to(torch.float64), atol=atol, rtol=0)


def check_corrector_disabled_passthrough(solve):
    c, p = solve(H3, MT, None, XT, SIG, 3, 1, 2)
    assert torch.equal(c, XT), c
    c2, p2 = solve(H3, MT, XT, XT, SIG, 3, 1, 2)
    assert not torch.equal(p, p2)
    # x_this 参与预测侧的 x 位；把它换掉结果必须移动
    c3, p3 = solve(H3, MT, None, torch.zeros_like(XT), SIG, 3, 1, 2)
    assert not torch.equal(p, p3)


def check_first_order_closed_form(solve):
    # order_p = 1 时 UniP 退化为 DDIM 闭式：prev = alpha_t*x0 + sigma_t*(x - alpha_s0*x0)/sigma_s0
    sig = [float(v) for v in SIG.tolist()]
    pair = lambda s: (1.0 / math.sqrt(s * s + 1.0), s / math.sqrt(s * s + 1.0))
    a_s0, s_s0 = pair(sig[3])
    a_t, s_t = pair(sig[4])
    x = XT.to(torch.float64)
    m0 = MT.to(torch.float64)
    want = a_t * m0 + s_t * (x - a_s0 * m0) / s_s0
    c, p = solve(H3, MT, None, XT, SIG, 3, 1, 1, True, "bh2", False)
    assert _close(p, torch.tensor(want, dtype=torch.float64).to(torch.float32)), (p, want)
    # bh1 / bh2 在没有差分项时不可能有别的结果
    _, p_bh1 = solve(H3, MT, None, XT, SIG, 3, 1, 1, True, "bh1", False)
    assert torch.equal(p, p_bh1)
    # predict_x0=False 的对应闭式：prev = (alpha_t/alpha_s0)*x + (sigma_t - alpha_t*sigma_s0/alpha_s0)*eps0
    want_eps = (a_t / a_s0) * x + (s_t - a_t * s_s0 / a_s0) * m0
    _, p_eps = solve(H3, MT, None, XT, SIG, 3, 1, 1, False, "bh2", False)
    assert _close(p_eps, torch.tensor(want_eps, dtype=torch.float64).to(torch.float32)), (p_eps, want_eps)


def check_solver_type_only_bites_on_simplified_rhos(solve):
    # 通式里 b 与 1/B_h 成正比，于是 B_h * rhos 与 solver_type 无关；
    # 只有被钉成 0.5 的那两支（预测侧 order_p == 2、校正侧 order_c == 1）会被 solver_type 移动。
    for oc, op in ((2, 3), (3, 3), (2, 1), (3, 1)):
        c1, p1 = solve(H3, MT, XL, XT, SIG, 3, oc, op, True, "bh1", False)
        c2, p2 = solve(H3, MT, XL, XT, SIG, 3, oc, op, True, "bh2", False)
        assert torch.equal(c1, c2) and torch.equal(p1, p2), (oc, op)
    # 校正侧 order_c == 1 钉住 0.5：校正值与继承它的预测值都会移动
    c1, p1 = solve(H3, MT, XL, XT, SIG, 3, 1, 3, True, "bh1", False)
    c2, p2 = solve(H3, MT, XL, XT, SIG, 3, 1, 3, True, "bh2", False)
    assert not torch.equal(c1, c2) and float((c1 - c2).abs().max()) > 0.05
    assert not torch.equal(p1, p2)
    # 预测侧 order_p == 2 钉住 0.5：只移动预测值，校正值仍逐位相同
    c3, p3 = solve(H3, MT, XL, XT, SIG, 3, 2, 2, True, "bh1", False)
    c4, p4 = solve(H3, MT, XL, XT, SIG, 3, 2, 2, True, "bh2", False)
    assert torch.equal(c3, c4)
    assert not torch.equal(p3, p4)
    assert float((p3 - p4).abs().max()) > 0.05
    # midpoint / heun / logrho 被静默映射成 bh2
    for alias in ("midpoint", "heun", "logrho"):
        ca, pa = solve(H3, MT, XL, XT, SIG, 3, 2, 2, True, alias, False)
        assert torch.equal(ca, c4) and torch.equal(pa, p4), alias


def check_predictor_order_two_uses_half(solve):
    # order_p == 2 时 rhos_p 恒为 0.5，可以按闭式手算
    sig = [float(v) for v in SIG.tolist()]
    pair = lambda s: (1.0 / math.sqrt(s * s + 1.0), s / math.sqrt(s * s + 1.0))
    a_s0, s_s0 = pair(sig[3])
    a_t, s_t = pair(sig[4])
    lam = lambda i: math.log(pair(sig[i])[0]) - math.log(pair(sig[i])[1])
    h = lam(4) - lam(3)
    hh = -h
    rk = (lam(2) - lam(3)) / h
    d1 = H3[-1].to(torch.float64) - MT.to(torch.float64)
    d1 = d1 / rk
    phi_1 = math.expm1(hh)
    b_h = math.expm1(hh)
    x = XT.to(torch.float64)
    m0 = MT.to(torch.float64)
    want = s_t / s_s0 * x - a_t * phi_1 * m0 - a_t * b_h * (0.5 * d1)
    c, p = solve(H3, MT, None, XT, SIG, 3, 1, 2, True, "bh2", False)
    assert _close(p, want.to(torch.float32), atol=5e-5), (p, want)
    assert math.isclose(rk, -1.0, rel_tol=1e-12)
    # 换成 bh1（B_h = hh）后这一支确实被移动，反证 0.5 不是通式解
    _, p_bh1 = solve(H3, MT, None, XT, SIG, 3, 1, 2, True, "bh1", False)
    assert not torch.equal(p, p_bh1)


def check_history_pairing_and_padding(solve):
    base_c, base_p = solve(H3, MT, XL, XT, SIG, 3, 2, 2)
    # 更老的历史项不参与本步，多塞一条必须逐位不变
    padded = (torch.randn_like(H3[0]),) + H3
    c, p = solve(padded, MT, XL, XT, SIG, 3, 2, 2)
    assert torch.equal(c, base_c) and torch.equal(p, base_p)
    # 相邻两项互换会移动结果：证明 m_hist[-1] 是上一步、m_hist[-2] 再上一步
    swapped = (H3[1], H3[2], H3[0])
    c2, p2 = solve(swapped, MT, XL, XT, SIG, 3, 2, 2)
    assert not torch.equal(p2, base_p)
    # 三阶需要 m_hist[-1] 与 m_hist[-2]（预测侧的 hist[-i]），与校正侧的 hist[-(i+1)] 不同源
    c3, p3 = solve(H3, MT, XL, XT, SIG, 3, 3, 3)
    c4, p4 = solve(H3, MT, XL, XT, SIG, 3, 2, 3)
    assert not torch.equal(c3, c4)


def check_flow_sigmas_branch(solve):
    sigf = torch.tensor([0.95, 0.8, 0.65, 0.5, 0.35, 0.2, 0.1], dtype=torch.float32)
    c, p = solve(H3, MT, XL, XT, sigf, 3, 2, 3, True, "bh2", True)
    assert torch.isfinite(c).all() and torch.isfinite(p).all()
    # flow 口径下 alpha = 1 - sigma，一阶预测有闭式
    s_s0, s_t = float(sigf[3]), float(sigf[4])
    a_s0, a_t = 1 - s_s0, 1 - s_t
    want = a_t * MT.to(torch.float64) + s_t * (XT.to(torch.float64) - a_s0 * MT.to(torch.float64)) / s_s0
    _, p1 = solve(H3, MT, None, XT, sigf, 3, 1, 1, True, "bh2", True)
    assert _close(p1, want.to(torch.float32)), (p1, want)
    # 同一 σ 数组按 VP 口径算，结果必然不同
    _, p_vp = solve(H3, MT, None, XT, sigf, 3, 1, 1, True, "bh2", False)
    assert not torch.equal(p1, p_vp)


def check_against_oracle(solve):
    g = torch.Generator().manual_seed(20261030)

    def ri(low, high):
        return int(torch.randint(low, high, (1,), generator=g).item())

    tried = 0
    for _ in range(240):
        b, cch = ri(1, 3), ri(1, 3)
        h, w = ri(1, 3), ri(1, 3)
        shape = (b, cch, h, w)
        T = ri(6, 11)
        flow = bool(ri(0, 2))
        if flow:
            base = [0.97 * (1.0 - i / (T + 2)) for i in range(T)]
        else:
            base = [12.0 / (i + 1) for i in range(T)]
        sig = torch.tensor(base, dtype=torch.float32)
        oc = ri(1, 4)
        op = ri(1, 4)
        lo = oc if ri(0, 2) else 1
        if lo == oc and T - 2 < oc:
            continue
        step = ri(oc if lo == oc else 1, T - 1)
        if step < max(oc if lo == oc else 0, op - 1):
            continue
        need = max(oc if lo == oc else 1, op - 1, 1)
        hist = tuple(torch.randn(shape, generator=g, dtype=torch.float32) for _ in range(need))
        mt = torch.randn(shape, generator=g, dtype=torch.float32)
        xt = torch.randn(shape, generator=g, dtype=torch.float32)
        if lo == oc:
            xl = torch.randn(shape, generator=g, dtype=torch.float32)
        else:
            xl = None
        px0 = bool(ri(0, 2))
        bh = "bh1" if ri(0, 2) else "bh2"
        tried += 1
        got = solve(hist, mt, xl, xt, sig, step, oc, op, px0, bh, flow)
        want = _oracle(hist, mt, xl, xt, sig, step, oc, op, px0, bh, flow, torch.Size(shape))
        for a, r, tag in ((got[0], want[0], "corr"), (got[1], want[1], "prev")):
            denom = max(1e-8, float(r.abs().max()))
            diff = float((a.to(torch.float64) - r).abs().max())
            assert diff / denom < 2e-4, (tag, diff, denom, (shape, oc, op, step, px0, bh, flow))
            assert torch.isfinite(a).all()
    assert tried > 120, tried


def check_dtypes_and_shapes(solve):
    hist64 = tuple(t.to(torch.float64) for t in H3)
    c, p = solve(hist64, MT.to(torch.float64), XL.to(torch.float64), XT.to(torch.float64),
                 SIG.to(torch.float64), 3, 2, 3)
    assert c.dtype == torch.float32 and p.dtype == torch.float32
    big = torch.randn(3, 2, 4, dtype=torch.float32)
    head = big[:, :1]
    hist_big = tuple(head + i for i in range(3))
    c2, p2 = solve(hist_big, head, big[:, 1:2], head, SIG, 3, 2, 2)
    assert tuple(c2.shape) == (3, 1, 4) and tuple(p2.shape) == (3, 1, 4)
    assert c2.dtype == torch.float32 and p2.dtype == torch.float32
    assert isinstance(solve(H3, MT, XL, XT, SIG, 3, 2, 2), tuple)


def check_invalid_inputs(solve):
    bad = [
        ("m_t 非张量", lambda: solve(H3, [[0.1]], XL, XT, SIG, 3, 1, 1)),
        ("m_hist 为空", lambda: solve((), MT, XL, XT, SIG, 3, 1, 1)),
        ("m_hist 非序列", lambda: solve(MT, MT, XL, XT, SIG, 3, 1, 1)),
        ("m_hist 含非张量", lambda: solve([[0.1]], MT, XL, XT, SIG, 3, 1, 1)),
        ("形状不一致", lambda: solve((torch.zeros(3, 2),), MT, XL, XT, SIG, 3, 1, 1)),
        ("一维张量", lambda: solve((torch.zeros(2),), torch.zeros(2), XL, XT, SIG, 3, 1, 1)),
        ("含 NaN", lambda: solve((torch.full((2, 2), float("nan")),), MT, XL, XT, SIG, 3, 1, 1)),
        ("含 inf", lambda: solve((torch.full((2, 2), float("inf")),), MT, XL, XT, SIG, 3, 1, 1)),
        ("x_this 非张量", lambda: solve(H3, MT, XL, [[0.0]], SIG, 3, 1, 1)),
        ("sigmas 非张量", lambda: solve(H3, MT, XL, XT, [8.0, 4.0], 1, 1, 1)),
        ("sigmas 二维", lambda: solve(H3, MT, XL, XT, SIG.reshape(7, 1), 3, 1, 1)),
        ("sigmas 含 0", lambda: solve(H3, MT, XL, XT, torch.tensor([8.0, 4.0, 2.0, 1.0, 0.0]), 3, 1, 1)),
        ("sigmas 含负数", lambda: solve(H3, MT, XL, XT, torch.tensor([8.0, 4.0, 2.0, -1.0, 0.5]), 3, 1, 1)),
        ("sigmas 含 NaN", lambda: solve(H3, MT, XL, XT, torch.tensor([8.0, float("nan"), 2.0, 1.0, 0.5]), 3, 1, 1)),
        ("step_index 越界", lambda: solve(H3, MT, XL, XT, SIG, 6, 1, 1)),
        ("step_index 非 int", lambda: solve(H3, MT, XL, XT, SIG, 3.0, 1, 1)),
        ("step_index 为 bool", lambda: solve(H3, MT, XL, XT, SIG, True, 1, 1)),
        ("step_index 为负", lambda: solve(H3, MT, XL, XT, SIG, -1, 1, 1)),
        ("order_c = 0", lambda: solve(H3, MT, XL, XT, SIG, 3, 0, 1)),
        ("order_p = 0", lambda: solve(H3, MT, XL, XT, SIG, 3, 1, 0)),
        ("order 为浮点", lambda: solve(H3, MT, XL, XT, SIG, 3, 1.0, 1)),
        ("order 为 bool", lambda: solve(H3, MT, XL, XT, SIG, 3, True, 1)),
        ("校正越界 step", lambda: solve(H3, MT, XL, XT, SIG, 1, 3, 1)),
        ("校正历史不足", lambda: solve(H3[:1], MT, XL, XT, SIG, 3, 2, 1)),
        ("预测历史不足", lambda: solve(H3[:1], MT, None, XT, SIG, 3, 1, 3)),
        ("predict_x0 非 bool", lambda: solve(H3, MT, XL, XT, SIG, 3, 1, 1, 1)),
        ("use_flow_sigmas 非 bool", lambda: solve(H3, MT, XL, XT, SIG, 3, 1, 1, True, "bh2", "yes")),
        ("solver_type 非字符串", lambda: solve(H3, MT, XL, XT, SIG, 3, 1, 1, True, 2)),
        ("solver_type 不认识", lambda: solve(H3, MT, XL, XT, SIG, 3, 1, 1, True, "dpmsolver")),
        ("flow 下 sigma >= 1", lambda: solve(H3, MT, XL, XT, SIG, 3, 1, 1, True, "bh2", True)),
    ]
    for name, fn in bad:
        try:
            fn()
        except ValueError:
            continue
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(f"{name}: 抛了 {type(exc).__name__}，应为 ValueError") from exc
        raise AssertionError(f"{name}: 未抛 ValueError")
