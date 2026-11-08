"""846 VITS normalizing flow 数值账 —— 判题用例。

四个 check 函数，均以 solve 为唯一入参（judge 内部 fn(solve) 调用）。
solve(mode, **kw) 分派：
  flow_primitive / coupling_layer / coupling_block / duration_nll
"""

import math

TOL = 1e-12
REL_TOL = 1e-9


def _close(a, b, tol=TOL):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def _cmp_scalar(got, want, label, tol=TOL):
    assert isinstance(got, (int, float)) and not isinstance(got, bool), (
        "%s: expected number, got %r" % (label, type(got))
    )
    assert _close(got, want, tol), "%s: got %r want %r" % (label, got, want)


def _cmp_list(got, want, label, tol=TOL):
    assert isinstance(got, list), "%s: expected list, got %r" % (label, type(got))
    assert len(got) == len(want), (
        "%s: length mismatch got %d want %d" % (label, len(got), len(want))
    )
    for i, (g, w) in enumerate(zip(got, want)):
        assert isinstance(g, (int, float)), "%s[%d]: expected number" % (label, i)
        assert _close(g, w, tol), "%s[%d]: got %r want %r" % (label, i, g, w)


def _expect_error(fn, label, exc=ValueError):
    try:
        fn()
    except exc:
        return
    except Exception as e:  # noqa: BLE001
        raise AssertionError(
            "%s: expected %s, got %s: %s"
            % (label, exc.__name__, type(e).__name__, str(e)[:200])
        )
    raise AssertionError("%s: expected %s, but no error raised" % (label, exc.__name__))


# --------------------------------------------------------------------------
# 固定测试数据
# --------------------------------------------------------------------------

# ---- Log flow ----
LOG_X1 = [[[0.5, 2.0, 4.0]]]
LOG_MASK1 = [[[1.0, 1.0, 1.0]]]
# clamp_min(x, 1e-5) 生效：三个值都小于 1e-5
LOG_X_CLAMP = [[[1e-8, 1e-5, 1e-6]]]
# 两个 batch，第二个 batch 的 t=1 被 mask 掉
LOG_X2 = [[[0.5, 2.0, 4.0]], [[1.0, 0.5, 0.25]]]
LOG_MASK2 = [[[1.0, 1.0, 1.0]], [[1.0, 0.0, 1.0]]]

# ---- Flip flow ----
FLIP_X = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]
FLIP_X4 = [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]]

# ---- ElementwiseAffine ----
AFF_X = [[[1.0, 2.0], [3.0, 4.0]]]
AFF_M = [0.5, -0.25]
AFF_LOGS = [0.25, -0.5]
AFF_M2 = [0.0, 0.1]
AFF_LOGS2 = [0.2, -0.3]

# ---- coupling ----
CP_X = [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]]
CP_MASK = [[[1.0, 1.0]]]
CP_M = [[[0.1, 0.2], [0.3, 0.4]]]
CP_LOGS = [[[0.5, -0.25], [0.125, 0.75]]]
# ★ mask=0 的位置：官方 `x1 = m + x1 * exp(logs) * x_mask` → 输出是 m（不是 0）
CP_X_MASKED = [
    [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
    [[-1.0, 0.5], [2.0, -3.0], [0.25, 0.75], [1.5, -0.25]],
]
CP_MASK_B = [[[1.0, 1.0]], [[1.0, 0.0]]]
CP_M_B = [
    [[0.1, 0.2], [0.3, 0.4]],
    [[0.0, -0.5], [0.25, 0.75]],
]
CP_LOGS_B = [
    [[0.5, -0.25], [0.125, 0.75]],
    [[0.0, 0.5], [-0.5, 0.25]],
]

# ---- duration ----
DUR_W = [[[2.0, 3.0, 4.0]], [[1.5, 2.5, 0.5]]]
DUR_EQ = [
    [[0.1, -0.2, 0.3], [0.4, -0.5, 0.6]],
    [[0.7, 0.8, -0.9], [1.0, 0.2, -0.3]],
]
DUR_ZU = [[[0.25, -0.5, 0.75]], [[0.1, 0.2, 0.3]]]
DUR_Z1 = [[[0.11, -0.22, 0.33]], [[0.44, -0.55, 0.66]]]
DUR_MASK = [[[1.0, 1.0, 1.0]], [[1.0, 0.0, 1.0]]]
DUR_POST_LD = [0.5, -0.25]
DUR_FLOW_LD = [1.25, 0.75]


# --------------------------------------------------------------------------
# 独立 oracle
# --------------------------------------------------------------------------


def _flt(v):
    if not isinstance(v, (list, tuple)):
        return [v]
    out = []
    for it in v:
        out.extend(_flt(it))
    return out


def _oracle_log(shape, x, mask, reverse=False):
    B, C, T = shape
    xs = [float(v) for v in _flt(x)]
    mk = [float(v) for v in _flt(mask)]
    if len(mk) == B * T:
        mk = [mk[b * T + t] for b in range(B) for c in range(C) for t in range(T)]
    out = [0.0] * (B * C * T)
    if reverse:
        for i in range(len(xs)):
            out[i] = math.exp(xs[i]) * mk[i]
        return out, [0.0] * B
    ld = [0.0] * B
    for b in range(B):
        acc = 0.0
        for c in range(C):
            for t in range(T):
                i = (b * C + c) * T + t
                v = xs[i]
                if v < 1e-5:
                    v = 1e-5
                y = math.log(v) * mk[i]
                out[i] = y
                acc += -y
        ld[b] = acc
    return out, ld


def _oracle_flip(shape, x):
    B, C, T = shape
    xs = [float(v) for v in _flt(x)]
    out = [0.0] * (B * C * T)
    for b in range(B):
        for c in range(C):
            for t in range(T):
                out[(b * C + c) * T + t] = xs[(b * C + (C - 1 - c)) * T + t]
    return out, [0.0] * B


def _oracle_affine(shape, x, mask, m, logs, reverse=False):
    B, C, T = shape
    xs = [float(v) for v in _flt(x)]
    mk = [float(v) for v in _flt(mask)]
    if len(mk) == B * T:
        mk = [mk[b * T + t] for b in range(B) for c in range(C) for t in range(T)]
    ms = [float(v) for v in _flt(m)]
    ls = [float(v) for v in _flt(logs)]
    out = [0.0] * (B * C * T)
    if reverse:
        for b in range(B):
            for c in range(C):
                s = math.exp(-ls[c])
                for t in range(T):
                    i = (b * C + c) * T + t
                    out[i] = (xs[i] - ms[c]) * s * mk[i]
        return out, [0.0] * B
    ld = [0.0] * B
    for b in range(B):
        acc = 0.0
        for c in range(C):
            s = math.exp(ls[c])
            for t in range(T):
                i = (b * C + c) * T + t
                out[i] = (ms[c] + s * xs[i]) * mk[i]
                acc += ls[c] * mk[i]
        ld[b] = acc
    return out, ld


def _oracle_coupling(shape, x, mask, m, logs=None, mean_only=True, reverse=False):
    B, C, T = shape
    half = C // 2
    xs = [float(v) for v in _flt(x)]
    mk = [float(v) for v in _flt(mask)]
    if len(mk) == B * T:
        mk = [mk[b * T + t] for b in range(B) for c in range(C) for t in range(T)]
    ms = [float(v) for v in _flt(m)]
    ls = [0.0] * len(ms) if mean_only else [float(v) for v in _flt(logs)]
    out = [0.0] * (B * C * T)
    ld = [0.0] * B
    for b in range(B):
        acc = 0.0
        for c in range(C):
            for t in range(T):
                i = (b * C + c) * T + t
                if c < half:
                    out[i] = xs[i]
                else:
                    j = b * half * T + (c - half) * T + t
                    if not reverse:
                        # ★ 官方：x1 = m + x1 * exp(logs) * x_mask
                        #   mask 只乘在 x1 项上，m 不被 mask
                        out[i] = ms[j] + math.exp(ls[j]) * xs[i] * mk[i]
                    else:
                        out[i] = (xs[i] - ms[j]) * math.exp(-ls[j]) * mk[i]
        if not mean_only:
            # logdet = sum(logs, [1,2])：★ 不乘 mask
            for j in range(b * half * T, b * half * T + half * T):
                acc += ls[j]
        ld[b] = acc
    return out, ld


def _oracle_block(shape, x, mask, m_list, reverse=False):
    B, C, T = shape
    steps = []
    for i in range(len(m_list)):
        steps.append(("coupling", i))
        steps.append(("flip", None))
    order = list(reversed(steps)) if reverse else steps
    cur = [float(v) for v in _flt(x)]
    ld = [0.0] * B
    for kind, idx in order:
        if kind == "flip":
            nxt = [0.0] * len(cur)
            for b in range(B):
                for c in range(C):
                    for t in range(T):
                        nxt[(b * C + c) * T + t] = cur[(b * C + (C - 1 - c)) * T + t]
            cur = nxt
            continue
        cur, sub = _oracle_coupling(
            (B, C, T), cur, mask, m_list[idx], mean_only=True, reverse=reverse
        )
        for b in range(B):
            ld[b] += sub[b]
    return cur, ld


def _oracle_duration(w, e_q, z_u, z1, mask, post_logdets, flow_logdets):
    wf = [float(v) for v in _flt(w)]
    eq = [float(v) for v in _flt(e_q)]
    zu = [float(v) for v in _flt(z_u)]
    z1v = [float(v) for v in _flt(z1)]
    mk = [float(v) for v in _flt(mask)]
    B = len(wf) // len(mk) if False else len(_flt(w))
    B = len(w)
    T = len(w[0][0])
    hl = 0.5 * math.log(2.0 * math.pi)

    def _sig(v):
        if v >= 0.0:
            return 1.0 / (1.0 + math.exp(-v))
        e = math.exp(v)
        return e / (1.0 + e)

    def _lsig(v):
        if v >= 0.0:
            return -math.log1p(math.exp(-v))
        return v - math.log1p(math.exp(v))

    logq = []
    for b in range(B):
        gauss = 0.0
        for c in range(2):
            for t in range(T):
                i = (b * 2 + c) * T + t
                mi = b * T + t
                gauss += -0.5 * (2.0 * hl + eq[i] ** 2) * mk[mi]
        lsig = 0.0
        for t in range(T):
            mi = b * T + t
            zv = zu[mi]
            lsig += (_lsig(zv) + _lsig(-zv)) * mk[mi]
        logq.append(gauss - (post_logdets[b] + lsig))

    z0 = []
    for b in range(B):
        for t in range(T):
            mi = b * T + t
            u = _sig(zu[mi]) * mk[mi]
            z0.append((wf[mi] - u) * mk[mi])

    z0_log = []
    logdet_log = [0.0] * B
    for b in range(B):
        for t in range(T):
            mi = b * T + t
            v = z0[b * T + t]
            if v < 1e-5:
                v = 1e-5
            y = math.log(v) * mk[mi]
            z0_log.append(y)
            logdet_log[b] += -y

    logdet_tot = [0.0] * B
    for b in range(B):
        logdet_tot[b] += logdet_log[b] + flow_logdets[b]

    nll = []
    for b in range(B):
        acc = 0.0
        for t in range(T):
            mi = b * T + t
            zi = z0_log[b * T + t]
            acc += 0.5 * (2.0 * hl + zi ** 2) * mk[mi]
            zi1 = z1v[b * T + t]
            acc += 0.5 * (2.0 * hl + zi1 ** 2) * mk[mi]
        nll.append(acc - logdet_tot[b])

    return {
        "logq": logq,
        "nll": nll,
        "total": [nll[b] + logq[b] for b in range(B)],
        "logdet_tot_q": [float(v) for v in post_logdets],
        "logdet_tot": logdet_tot,
        "z0": z0,
        "z0_log": z0_log,
    }


# --------------------------------------------------------------------------
# check 1: flow_primitive
# --------------------------------------------------------------------------


def check_flow_primitive(solve):
    # ---- Log ----
    r = solve(
        "flow_primitive", kind="log", shape=[1, 1, 3],
        x=LOG_X1, mask=LOG_MASK1,
    )
    oy, old = _oracle_log([1, 1, 3], LOG_X1, LOG_MASK1)
    _cmp_list(r["y"], oy, "log.y")
    _cmp_list(r["logdet"], old, "log.logdet")
    # logdet = sum(-y)
    _cmp_scalar(r["logdet"][0], -sum(oy), "log.logdet.sum")

    # ---- Log：clamp_min(x, 1e-5) 生效 ----
    r2 = solve(
        "flow_primitive", kind="log", shape=[1, 1, 3],
        x=LOG_X_CLAMP, mask=LOG_MASK1,
    )
    oy2, old2 = _oracle_log([1, 1, 3], LOG_X_CLAMP, LOG_MASK1)
    _cmp_list(r2["y"], oy2, "log.clamp.y")
    _cmp_list(r2["logdet"], old2, "log.clamp.logdet")
    # 三个位置都被 clamp 到 1e-5 → y 全是 log(1e-5)
    _cmp_scalar(r2["y"][0], math.log(1e-5), "log.clamp.y0")
    _cmp_scalar(r2["y"][2], math.log(1e-5), "log.clamp.y2")

    # ---- Log：mask=0 的位置 y=0 且不贡献 logdet ----
    r3 = solve(
        "flow_primitive", kind="log", shape=[2, 1, 3],
        x=LOG_X2, mask=LOG_MASK2,
    )
    oy3, old3 = _oracle_log([2, 1, 3], LOG_X2, LOG_MASK2)
    _cmp_list(r3["y"], oy3, "log.mask.y")
    _cmp_list(r3["logdet"], old3, "log.mask.logdet")
    _cmp_scalar(r3["y"][3], 0.0, "log.mask.y3")

    # ---- Log：reverse（输入是 forward 的输出）----
    r4 = solve(
        "flow_primitive", kind="log", shape=[1, 1, 3],
        x=LOG_X1, mask=LOG_MASK1, reverse=True,
    )
    oy4, old4 = _oracle_log([1, 1, 3], LOG_X1, LOG_MASK1, reverse=True)
    _cmp_list(r4["y"], oy4, "log.rev.y")
    # reverse 不返回 logdet
    assert r4["logdet"] == [0.0], "log.rev.logdet"

    # ---- Log：★ reverse 也要乘 mask（官方 `x = torch.exp(x) * x_mask`）----
    r4b = solve(
        "flow_primitive", kind="log", shape=[2, 1, 3],
        x=LOG_X2, mask=LOG_MASK2, reverse=True,
    )
    oy4b, old4b = _oracle_log([2, 1, 3], LOG_X2, LOG_MASK2, reverse=True)
    _cmp_list(r4b["y"], oy4b, "log.rev.mask.y")
    _cmp_list(r4b["logdet"], old4b, "log.rev.mask.logdet")
    # 第 2 个 batch 的 t=1：mask=0 → exp(x) * 0 = 0
    _cmp_scalar(r4b["y"][4], 0.0, "log.rev.mask.y4")
    # 第 2 个 batch 的 t=0：mask=1 → exp(1.0)
    _cmp_scalar(r4b["y"][3], math.exp(1.0), "log.rev.mask.y3")

    # ---- Flip：★ 翻的是 channel 轴 ----
    r5 = solve(
        "flow_primitive", kind="flip", shape=[1, 2, 3],
        x=FLIP_X, mask=LOG_MASK1,
    )
    oy5, old5 = _oracle_flip([1, 2, 3], FLIP_X)
    _cmp_list(r5["y"], oy5, "flip.y")
    _cmp_list(r5["logdet"], old5, "flip.logdet")
    # 显式断言：2 channel 的 flip = 交换两半（不是翻转时间轴）
    _cmp_scalar(r5["y"][0], 4.0, "flip.y0")
    _cmp_scalar(r5["y"][2], 6.0, "flip.y2")
    _cmp_scalar(r5["y"][3], 1.0, "flip.y3")
    _cmp_scalar(r5["y"][5], 3.0, "flip.y5")

    # ---- Flip：4 channel ----
    r6 = solve(
        "flow_primitive", kind="flip", shape=[1, 4, 2],
        x=FLIP_X4, mask=[[[1.0, 1.0]]],
    )
    oy6, old6 = _oracle_flip([1, 4, 2], FLIP_X4)
    _cmp_list(r6["y"], oy6, "flip4.y")
    _cmp_scalar(r6["y"][0], 7.0, "flip4.y0")
    _cmp_scalar(r6["y"][6], 1.0, "flip4.y6")

    # ---- ★ Flip 忽略 x_mask（官方 Flip.forward 完全没有 mask 参数）----
    r6b = solve(
        "flow_primitive", kind="flip", shape=[2, 2, 2],
        x=[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
        mask=[[[1.0, 0.0]], [[0.0, 1.0]]],
    )
    oy6b, old6b = _oracle_flip(
        [2, 2, 2], [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    _cmp_list(r6b["y"], oy6b, "flip.mask.y")
    _cmp_list(r6b["logdet"], old6b, "flip.mask.logdet")
    # batch0 的 t=1 被 mask 掉，但 Flip 仍然照常交换 channel
    _cmp_scalar(r6b["y"][0], 3.0, "flip.mask.y0")
    _cmp_scalar(r6b["y"][2], 1.0, "flip.mask.y2")
    _cmp_scalar(r6b["y"][4], 7.0, "flip.mask.y4")
    _cmp_scalar(r6b["y"][6], 5.0, "flip.mask.y6")

    # ---- ElementwiseAffine ----
    r7 = solve(
        "flow_primitive", kind="affine", shape=[1, 2, 2],
        x=AFF_X, mask=[[[1.0, 1.0]]], m=AFF_M, logs=AFF_LOGS,
    )
    oy7, old7 = _oracle_affine([1, 2, 2], AFF_X, [[[1.0, 1.0]]], AFF_M, AFF_LOGS)
    _cmp_list(r7["y"], oy7, "aff.y")
    _cmp_list(r7["logdet"], old7, "aff.logdet")
    # logdet = sum(logs * mask)，两个 channel 各 2 个位置
    _cmp_scalar(r7["logdet"][0], 2 * (AFF_LOGS[0] + AFF_LOGS[1]), "aff.logdet.sum")

    # ---- ElementwiseAffine：reverse（输入是 forward 的输出）----
    r7f = solve(
        "flow_primitive", kind="affine", shape=[1, 2, 2],
        x=AFF_X, mask=[[[1.0, 1.0]]], m=AFF_M, logs=AFF_LOGS,
    )
    r8 = solve(
        "flow_primitive", kind="affine", shape=[1, 2, 2],
        x=r7f["y"], mask=[[[1.0, 1.0]]], m=AFF_M, logs=AFF_LOGS, reverse=True,
    )
    oy8, old8 = _oracle_affine(
        [1, 2, 2], r7f["y"], [[[1.0, 1.0]]], AFF_M, AFF_LOGS, reverse=True)
    _cmp_list(r8["y"], oy8, "aff.rev.y")
    # forward → reverse 回到原 x
    _cmp_list(r8["y"], [1.0, 2.0, 3.0, 4.0], "aff.roundtrip")

    # ---- ElementwiseAffine：mask=0 的位置输出 0（整体 mask）----
    r9 = solve(
        "flow_primitive", kind="affine", shape=[2, 2, 2],
        x=[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
        mask=[[[1.0, 1.0]], [[1.0, 0.0]]],
        m=AFF_M2, logs=AFF_LOGS2,
    )
    oy9, old9 = _oracle_affine(
        [2, 2, 2],
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
        [[[1.0, 1.0]], [[1.0, 0.0]]], AFF_M2, AFF_LOGS2,
    )
    _cmp_list(r9["y"], oy9, "aff.mask.y")
    _cmp_list(r9["logdet"], old9, "aff.mask.logdet")
    # 第 2 个 batch 的 t=1：y = (m + exp(logs)*x) * 0 = 0
    _cmp_scalar(r9["y"][5], 0.0, "aff.mask.y5")
    _cmp_scalar(r9["y"][7], 0.0, "aff.mask.y7")

    # ---- 非法输入 ----
    _expect_error(
        lambda: solve(
            "flow_primitive", kind="nope", shape=[1, 1, 2],
            x=[[[1.0, 2.0]]], mask=[[[1.0, 1.0]]],
        ),
        "fp.unknown-kind",
    )
    _expect_error(
        lambda: solve(
            "flow_primitive", kind="affine", shape=[1, 1, 2],
            x=[[[1.0, 2.0]]], mask=[[[1.0, 1.0]]],
        ),
        "fp.affine-missing-m",
    )
    _expect_error(
        lambda: solve(
            "flow_primitive", kind="affine", shape=[1, 2, 2],
            x=AFF_X, mask=[[[1.0, 1.0]]], m=[0.5, 0.1, 0.2], logs=AFF_LOGS,
        ),
        "fp.affine-m-wrong-len",
    )
    _expect_error(
        lambda: solve(
            "flow_primitive", kind="log", shape=[1, 1, 2],
            x=[[[1.0, 2.0, 3.0]]], mask=[[[1.0, 1.0]]],
        ),
        "fp.x-length",
    )
    _expect_error(
        lambda: solve(
            "flow_primitive", kind="log", shape=[1, 1, 0],
            x=[[[1.0, 2.0]]], mask=[[[1.0, 1.0]]],
        ),
        "fp.shape-nonpositive",
    )
    _expect_error(
        lambda: solve(
            "flow_primitive", kind="log", shape=[1, 1],
            x=[[[1.0, 2.0]]], mask=[[[1.0, 1.0]]],
        ),
        "fp.shape-arity",
    )


# --------------------------------------------------------------------------
# check 2: coupling_layer
# --------------------------------------------------------------------------


def check_coupling_layer(solve):
    # ---- mean_only=True：logs 恒 0 → logdet 恒 0 ----
    r = solve(
        "coupling_layer", shape=[1, 4, 2], x=CP_X, mask=CP_MASK, m=CP_M,
        mean_only=True,
    )
    oy, old = _oracle_coupling([1, 4, 2], CP_X, CP_MASK, CP_M, mean_only=True)
    _cmp_list(r["y"], oy, "cp.mo.y")
    _cmp_list(r["logdet"], old, "cp.mo.logdet")
    _cmp_scalar(r["logdet"][0], 0.0, "cp.mo.logdet0")

    # ---- mean_only=False：logdet = sum(logs) ----
    r2 = solve(
        "coupling_layer", shape=[1, 4, 2], x=CP_X, mask=CP_MASK, m=CP_M,
        logs=CP_LOGS, mean_only=False,
    )
    oy2, old2 = _oracle_coupling(
        [1, 4, 2], CP_X, CP_MASK, CP_M, CP_LOGS, mean_only=False)
    _cmp_list(r2["y"], oy2, "cp.logs.y")
    _cmp_list(r2["logdet"], old2, "cp.logs.logdet")
    # sum(logs) = 0.5 - 0.25 + 0.125 + 0.75
    _cmp_scalar(r2["logdet"][0], 0.5 - 0.25 + 0.125 + 0.75, "cp.logs.logdet0")

    # ---- ★ mask=0 的位置输出是 m（不是 0）----
    r3 = solve(
        "coupling_layer", shape=[2, 4, 2], x=CP_X_MASKED, mask=CP_MASK_B,
        m=CP_M_B, logs=CP_LOGS_B, mean_only=False,
    )
    oy3, old3 = _oracle_coupling(
        [2,  4, 2], CP_X_MASKED, CP_MASK_B, CP_M_B, CP_LOGS_B, mean_only=False)
    _cmp_list(r3["y"], oy3, "cp.mask.y")
    _cmp_list(r3["logdet"], old3, "cp.mask.logdet")
    # 第 2 个 batch 的 t=1：mask=0 → 输出 = m
    #   m 的 channel 2 (x1 的第 0 个) t=1 → -0.5
    #   m 的 channel 3 (x1 的第 1 个) t=1 → 0.75
    _cmp_scalar(r3["y"][13], -0.5, "cp.mask.y13")
    _cmp_scalar(r3["y"][15], 0.75, "cp.mask.y15")
    # ★ logdet 不乘 mask：第 2 个 batch 的 logdet 仍含被 mask 位置的 logs
    _cmp_scalar(r3["logdet"][1], 0.0 + 0.5 - 0.5 + 0.25, "cp.mask.logdet1")

    # ---- reverse：应回到原 x（mask 全 1 时）----
    r4 = solve(
        "coupling_layer", shape=[1, 4, 2], x=CP_X, mask=CP_MASK, m=CP_M,
        logs=CP_LOGS, mean_only=False, reverse=True,
    )
    oy4, old4 = _oracle_coupling(
        [1, 4, 2], CP_X, CP_MASK, CP_M, CP_LOGS, mean_only=False, reverse=True)
    _cmp_list(r4["y"], oy4, "cp.rev.y")
    # 先 forward 再 reverse → 回到原 x
    fwd = solve(
        "coupling_layer", shape=[1, 4, 2], x=CP_X, mask=CP_MASK, m=CP_M,
        logs=CP_LOGS, mean_only=False,
    )
    back = solve(
        "coupling_layer", shape=[1, 4, 2], x=fwd["y"], mask=CP_MASK, m=CP_M,
        logs=CP_LOGS, mean_only=False, reverse=True,
    )
    _cmp_list(back["y"], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], "cp.roundtrip")

    # ---- 非法输入 ----
    # channels 必须是偶数
    _expect_error(
        lambda: solve(
            "coupling_layer", shape=[1, 3, 2],
            x=[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]],
            mask=[[[1.0, 1.0]]], m=[[[0.1, 0.2]]],
        ),
        "cp.odd-channels",
    )
    # m 的长度必须是 B * (C/2) * T
    _expect_error(
        lambda: solve(
            "coupling_layer", shape=[1, 4, 2], x=CP_X, mask=CP_MASK,
            m=[[[0.1, 0.2]]],
        ),
        "cp.m-wrong-len",
    )
    # mean_only=False 但没给 logs
    _expect_error(
        lambda: solve(
            "coupling_layer", shape=[1, 4, 2], x=CP_X, mask=CP_MASK, m=CP_M,
            mean_only=False,
        ),
        "cp.logs-missing",
    )
    # logs 长度与 m 不一致
    _expect_error(
        lambda: solve(
            "coupling_layer", shape=[1, 4, 2], x=CP_X, mask=CP_MASK, m=CP_M,
            logs=[[[0.5, -0.25]]], mean_only=False,
        ),
        "cp.logs-wrong-len",
    )
    # 缺 m
    _expect_error(
        lambda: solve(
            "coupling_layer", shape=[1, 4, 2], x=CP_X, mask=CP_MASK,
        ),
        "cp.m-missing",
    )


# --------------------------------------------------------------------------
# check 3: coupling_block
# --------------------------------------------------------------------------


def check_coupling_block(solve):
    m_list1 = [[[[0.5, 0.25]]]]
    m_list2 = [
        [[[0.1, 0.2], [0.3, 0.4]]],
        [[[0.5, -0.25], [0.125, 0.75]]],
    ]

    # ---- 单个 coupling + flip ----
    r = solve(
        "coupling_block", shape=[1, 2, 2], x=[[[1.0, 2.0], [3.0, 4.0]]],
        mask=[[[1.0, 1.0]]], m_list=m_list1,
    )
    oy, old = _oracle_block(
        [1, 2, 2], [[[1.0, 2.0], [3.0, 4.0]]], [[[1.0, 1.0]]], m_list1)
    _cmp_list(r["y"], oy, "blk.y")
    _cmp_list(r["logdet"], old, "blk.logdet")
    # 每个 coupling 都是 mean_only=True → logdet 恒 0
    _cmp_scalar(r["logdet"][0], 0.0, "blk.logdet0")
    # 显式：coupling 后 flip（2 channel → 交换两半）
    #   coupling: x0=[1,2] 原样；x1=[3,4] → m + x1*1 = [3.5, 4.25]
    #   flip: 交换两半 → [[3.5,4.25],[1,2]]
    _cmp_scalar(r["y"][0], 3.5, "blk.y0")
    _cmp_scalar(r["y"][2], 1.0, "blk.y2")

    # ---- 两个 coupling + 两个 flip ----
    r2 = solve(
        "coupling_block", shape=[1, 4, 2], x=CP_X, mask=CP_MASK, m_list=m_list2,
    )
    oy2, old2 = _oracle_block([1, 4, 2], CP_X, CP_MASK, m_list2)
    _cmp_list(r2["y"], oy2, "blk2.y")
    _cmp_list(r2["logdet"], old2, "blk2.logdet")

    # ---- reverse：逆序执行（官方 `for flow in reversed(self.flows)`）----
    r3 = solve(
        "coupling_block", shape=[1, 4, 2], x=CP_X, mask=CP_MASK, m_list=m_list2,
        reverse=True,
    )
    oy3, old3 = _oracle_block(
        [1, 4, 2], CP_X, CP_MASK, m_list2, reverse=True)
    _cmp_list(r3["y"], oy3, "blk2.rev.y")

    # ---- ★ reverse 的顺序必须可观测：单个 coupling 的 block ----
    # 官方 flows = [Coupling_0, Flip]；reverse 时逆序执行 -> [Flip, Coupling_0]
    # 若误按正序 [Coupling_0, Flip] 执行，结果不同
    r3b = solve(
        "coupling_block", shape=[1, 2, 2], x=[[[1.0, 2.0], [3.0, 4.0]]],
        mask=[[[1.0, 1.0]]], m_list=m_list1, reverse=True,
    )
    oy3b, old3b = _oracle_block(
        [1, 2, 2], [[[1.0, 2.0], [3.0, 4.0]]], [[[1.0, 1.0]]], m_list1,
        reverse=True,
    )
    _cmp_list(r3b["y"], oy3b, "blk1.rev.y")
    # 显式断言：逆序 = Flip -> Coupling(rev)
    #   x = [[1,2],[3,4]]；Flip -> [[3,4],[1,2]]；Coupling(rev): x1 -= m
    #   -> [[3,4],[1-0.5, 2-0.25]] = [[3,4],[0.5,1.75]]
    _cmp_scalar(r3b["y"][0], 3.0, "blk1.rev.y0")
    _cmp_scalar(r3b["y"][2], 0.5, "blk1.rev.y2")
    _cmp_scalar(r3b["y"][3], 1.75, "blk1.rev.y3")
    # 若按正序 [Coupling(rev), Flip] 执行，y0 会是 2.5 而不是 3.0
    assert abs(r3b["y"][0] - 3.0) < 1e-12, "blk1.rev.order"

    # ---- forward → reverse 回到原 x（mask 全 1）----
    fwd = solve(
        "coupling_block", shape=[1, 4, 2], x=CP_X, mask=CP_MASK, m_list=m_list2,
    )
    back = solve(
        "coupling_block", shape=[1, 4, 2], x=fwd["y"], mask=CP_MASK,
        m_list=m_list2, reverse=True,
    )
    _cmp_list(back["y"], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], "blk.roundtrip")

    # ---- 非法输入 ----
    _expect_error(
        lambda: solve(
            "coupling_block", shape=[1, 2, 2], x=[[[1.0, 2.0], [3.0, 4.0]]],
            mask=[[[1.0, 1.0]]], m_list=[],
        ),
        "blk.empty-m-list",
    )
    _expect_error(
        lambda: solve(
            "coupling_block", shape=[1, 3, 2],
            x=[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]],
            mask=[[[1.0, 1.0]]], m_list=m_list1,
        ),
        "blk.odd-channels",
    )
    _expect_error(
        lambda: solve(
            "coupling_block", shape=[1, 2, 2], x=[[[1.0, 2.0], [3.0, 4.0]]],
            mask=[[[1.0, 1.0]]], m_list=[[[[0.1, 0.2, 0.3]]]],
        ),
        "blk.m-wrong-len",
    )


# --------------------------------------------------------------------------
# check 4: duration_nll
# --------------------------------------------------------------------------


def check_duration_nll(solve):
    r = solve(
        "duration_nll", w=DUR_W, e_q=DUR_EQ, z_u=DUR_ZU, z1=DUR_Z1,
        mask=DUR_MASK, post_logdets=DUR_POST_LD, flow_logdets=DUR_FLOW_LD,
    )
    o = _oracle_duration(
        DUR_W, DUR_EQ, DUR_ZU, DUR_Z1, DUR_MASK, DUR_POST_LD, DUR_FLOW_LD)
    _cmp_list(r["logq"], o["logq"], "dur.logq")
    _cmp_list(r["nll"], o["nll"], "dur.nll")
    _cmp_list(r["total"], o["total"], "dur.total")
    _cmp_list(r["logdet_tot"], o["logdet_tot"], "dur.logdet_tot")
    _cmp_list(r["logdet_tot_q"], o["logdet_tot_q"], "dur.logdet_tot_q")
    _cmp_list(r["z0"], o["z0"], "dur.z0")
    _cmp_list(r["z0_log"], o["z0_log"], "dur.z0_log")
    # total = nll + logq
    for i in range(len(r["total"])):
        _cmp_scalar(r["total"][i], r["nll"][i] + r["logq"][i], "dur.total[%d]" % i)

    # ★ logq 的 logsigmoid 项是被【减去】的（它本身是负数）
    #   logq = gauss - (post_logdets + lsig)
    #   → logq > gauss - post_logdets
    gauss = []
    hl = 0.5 * math.log(2.0 * math.pi)
    for b in range(2):
        acc = 0.0
        for c in range(2):
            for t in range(3):
                i = (b * 2 + c) * 3 + t
                mk = DUR_MASK[b][0][t]
                acc += -0.5 * (2.0 * hl + DUR_EQ[b][c][t] ** 2) * mk
        gauss.append(acc)
    for b in range(2):
        assert r["logq"][b] > gauss[b] - DUR_POST_LD[b] - 1e-12, (
            "dur.logq.sign[%d]" % b
        )

    # ★ z0 = (w - sigmoid(z_u) * mask) * mask
    for b in range(2):
        for t in range(3):
            mk = DUR_MASK[b][0][t]
            u = 1.0 / (1.0 + math.exp(-DUR_ZU[b][0][t]))
            want = (DUR_W[b][0][t] - u * mk) * mk
            _cmp_scalar(r["z0"][b * 3 + t], want, "dur.z0[%d][%d]" % (b, t))

    # ★ z0_log 用 clamp_min(1e-5)
    for b in range(2):
        for t in range(3):
            v = r["z0"][b * 3 + t]
            if v < 1e-5:
                want = math.log(1e-5) * DUR_MASK[b][0][t]
            else:
                want = math.log(v) * DUR_MASK[b][0][t]
            _cmp_scalar(r["z0_log"][b * 3 + t], want, "dur.z0_log[%d][%d]" % (b, t))

    # ---- 非法输入 ----
    _expect_error(
        lambda: solve(
            "duration_nll", w=DUR_W, e_q=DUR_EQ, z_u=DUR_ZU, z1=DUR_Z1,
            mask=DUR_MASK, post_logdets=[0.5], flow_logdets=DUR_FLOW_LD,
        ),
        "dur.post-logdets-len",
    )
    _expect_error(
        lambda: solve(
            "duration_nll", w=DUR_W, e_q=DUR_EQ, z_u=DUR_ZU, z1=DUR_Z1,
            mask=DUR_MASK, post_logdets=DUR_POST_LD, flow_logdets=[1.25],
        ),
        "dur.flow-logdets-len",
    )
    _expect_error(
        lambda: solve(
            "duration_nll", w=DUR_W, e_q=[[[0.1, 0.2, 0.3]]], z_u=DUR_ZU,
            z1=DUR_Z1, mask=DUR_MASK, post_logdets=DUR_POST_LD,
            flow_logdets=DUR_FLOW_LD,
        ),
        "dur.e-q-shape",
    )
    _expect_error(
        lambda: solve(
            "duration_nll", w=[[[2.0, 3.0]]], e_q=DUR_EQ, z_u=DUR_ZU,
            z1=DUR_Z1, mask=DUR_MASK, post_logdets=DUR_POST_LD,
            flow_logdets=DUR_FLOW_LD,
        ),
        "dur.w-shape",
    )
    _expect_error(
        lambda: solve(
            "duration_nll", w=DUR_W, e_q=DUR_EQ, z_u=DUR_ZU, z1=DUR_Z1,
            mask=[[[1.0, 1.0, 1.0]]], post_logdets=DUR_POST_LD,
            flow_logdets=DUR_FLOW_LD,
        ),
        "dur.mask-shape",
    )
