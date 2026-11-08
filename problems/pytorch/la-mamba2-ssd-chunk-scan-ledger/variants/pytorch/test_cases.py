"""850 判题用例：Mamba2 SSD chunk 扫描三段账。

四个 check_*：
  check_chunk_state       —— chunk 内状态账
  check_state_passing     —— chunk 间状态传递账
  check_chunk_scan        —— chunk 内输出账
  check_ssd_forward       —— 三段完整编排账

每个 check 都带独立 oracle（按官方算式重写的纯 Python 实现），
并对关键结论做**双向断言**（等于 oracle + 必须不等于那个「看起来也合理」的错误候选）。
"""

import math


# ----------------------------------------------------------------- 工具
def _get(res, key, default=None):
    if isinstance(res, dict):
        if key in res:
            return res[key]
        return default
    return default


def _flat(x):
    out = []
    stack = [x]
    while stack:
        node = stack.pop(0)
        if isinstance(node, (list, tuple)):
            stack = list(node) + stack
        else:
            out.append(float(node))
    return out


def _close(a, b, tol=1e-9):
    fa, fb = _flat(a), _flat(b)
    if len(fa) != len(fb):
        return False
    for u, v in zip(fa, fb):
        if abs(u - v) > tol:
            return False
    return True


def _must_differ(got, wrong, msg):
    """双向断言：got 必须不等于那个错误候选。"""
    if _close(got, wrong, tol=1e-9):
        raise AssertionError(msg)


def _cumsum(vals):
    out = []
    acc = 0.0
    for v in vals:
        acc += v
        out.append(acc)
    return out


# --------------------------------------------------------------- oracle 层
def _oracle_chunk_state(x, dt, A, B_mat, chunk_size):
    """按官方 chunk_state_ref 算式独立重写。"""
    b = len(x)
    l = len(x[0])
    h = len(x[0][0])
    p = len(x[0][0][0])
    g = len(B_mat[0][0])
    n = len(B_mat[0][0][0])
    c = l // chunk_size
    lc = chunk_size
    hh = h // g

    expB = [[[[B_mat[bi][li][gi][ni] for ni in range(n)]
              for gi in range(g) for _ in range(hh)]
             for li in range(l)] for bi in range(b)]

    dAcs = [[[_cumsum([dt[bi][hi][ci][li] * A[hi] for li in range(lc)])
              for ci in range(c)] for hi in range(h)] for bi in range(b)]
    dec = [[[[math.exp(dAcs[bi][hi][ci][lc - 1] - dAcs[bi][hi][ci][li])
              for li in range(lc)] for ci in range(c)] for hi in range(h)]
            for bi in range(b)]

    states = []
    for bi in range(b):
        pc = []
        for ci in range(c):
            ph = []
            for hi in range(h):
                pp = []
                for pi in range(p):
                    pn = []
                    for ni in range(n):
                        s = 0.0
                        for li in range(lc):
                            s += (expB[bi][ci * lc + li][hi][ni]
                                  * dec[bi][hi][ci][li]
                                  * dt[bi][hi][ci][li]
                                  * x[bi][ci * lc + li][hi][pi])
                        pn.append(s)
                    pp.append(pn)
                ph.append(pp)
            pc.append(ph)
        states.append(pc)
    return {"dA_cumsum": dAcs, "decay_states": dec, "states": states,
            "expanded_B": expB}


def _oracle_state_passing(states, dAcs_chunk):
    b = len(states)
    c = len(states[0])
    h = len(states[0][0])
    d = len(states[0][0][0])
    dacc = []
    for bi in range(b):
        ph = []
        for hi in range(h):
            ph.append(_cumsum([0.0] + [dAcs_chunk[bi][hi][ci] for ci in range(c)]))
        dacc.append(ph)
    nc = c + 1
    dec = []
    for bi in range(b):
        ph = []
        for hi in range(h):
            pz = []
            for zi in range(nc):
                row = []
                for zj in range(nc):
                    row.append(math.exp(dacc[bi][hi][zi] - dacc[bi][hi][zj])
                               if zj <= zi else 0.0)
                pz.append(row)
            ph.append(pz)
        dec.append(ph)
    cat = [[[[0.0] * d for _ in range(h)]] + states[bi] for bi in range(b)]
    out = []
    for bi in range(b):
        pz = []
        for zi in range(nc):
            ph = []
            for hi in range(h):
                pd = []
                for di in range(d):
                    s = 0.0
                    for zc in range(nc):
                        s += dec[bi][hi][zi][zc] * cat[bi][zc][hi][di]
                    pd.append(s)
                ph.append(pd)
            pz.append(ph)
        out.append(pz)
    return {"decay_chunk": dec, "passed": [out[bi][:c] for bi in range(b)],
            "final_state": [out[bi][nc - 1] for bi in range(b)]}


def _oracle_chunk_scan(B_mat, C_mat, x, dt, dA_cumsum, prev_states, D=None, z=None):
    b = len(x)
    l = len(x[0])
    h = len(x[0][0])
    p = len(x[0][0][0])
    g = len(B_mat[0][0])
    n = len(B_mat[0][0][0])
    c = len(dt[0][0])
    lc = len(dt[0][0][0])
    hh = h // g

    def expand(mat):
        return [[[[mat[bi][li][gi][ni] for ni in range(n)]
                  for gi in range(g) for _ in range(hh)]
                 for li in range(l)] for bi in range(b)]

    B_exp = expand(B_mat)
    C_exp = expand(C_mat)

    CB = []
    for bi in range(b):
        pc = []
        for ci in range(c):
            ph = []
            for hi in range(h):
                pl = []
                for li in range(lc):
                    row = []
                    for si in range(lc):
                        s = 0.0
                        for ni in range(n):
                            s += (C_exp[bi][ci * lc + li][hi][ni]
                                  * B_exp[bi][ci * lc + si][hi][ni])
                        row.append(s)
                    pl.append(row)
                ph.append(pl)
            pc.append(ph)
        CB.append(pc)

    dec = []
    for bi in range(b):
        ph = []
        for hi in range(h):
            pc = []
            for ci in range(c):
                pl = []
                for li in range(lc):
                    row = []
                    for si in range(lc):
                        v = dA_cumsum[bi][hi][ci][li] - dA_cumsum[bi][hi][ci][si]
                        row.append(math.exp(v) if v < 700.0 else float("inf"))
                    pl.append(row)
                pc.append(pl)
            ph.append(pc)
        dec.append(ph)

    sd = []
    for bi in range(b):
        pc = []
        for ci in range(c):
            ph = []
            for hi in range(h):
                pl = []
                for li in range(lc):
                    row = []
                    for si in range(lc):
                        v = CB[bi][ci][hi][li][si] * dec[bi][hi][ci][li][si]
                        row.append(v if si <= li else 0.0)
                    pl.append(row)
                ph.append(pl)
            pc.append(ph)
        sd.append(pc)

    out = []
    for bi in range(b):
        pl_all = []
        for li in range(l):
            ci = li // lc
            sub = li % lc
            ph = []
            for hi in range(h):
                row = []
                for pi in range(p):
                    intra = 0.0
                    for si in range(lc):
                        intra += (sd[bi][ci][hi][sub][si]
                                  * dt[bi][hi][ci][si]
                                  * x[bi][ci * lc + si][hi][pi])
                    prev_c = 0.0
                    for ni in range(n):
                        prev_c += (C_exp[bi][ci * lc + sub][hi][ni]
                                   * prev_states[bi][ci][hi][pi][ni])
                    sd_exp = math.exp(dA_cumsum[bi][hi][ci][sub])
                    row.append(intra + prev_c * sd_exp)
                ph.append(row)
            pl_all.append(ph)
        out.append(pl_all)

    if D is not None:
        for bi in range(b):
            for li in range(l):
                for hi in range(h):
                    for pi in range(p):
                        if isinstance(D[0], (list, tuple)):
                            dv = D[hi][pi]
                        else:
                            dv = D[hi]
                        out[bi][li][hi][pi] += x[bi][li][hi][pi] * dv
    if z is not None:
        for bi in range(b):
            for li in range(l):
                for hi in range(h):
                    for pi in range(p):
                        zv = z[bi][li][hi][pi]
                        silu = zv / (1.0 + math.exp(-zv)) if zv >= -700.0 else 0.0
                        out[bi][li][hi][pi] *= silu
    return {"CB": CB, "scores_decay": sd, "out": out}


# ------------------------------------------------------------ 用例数据
def _mk_case(b, l, h, p, g, n, chunk, seed=0):
    import random
    rnd = random.Random(seed)
    def rr(shape, lo=-1.0, hi=1.0):
        if not shape:
            return rnd.uniform(lo, hi)
        return [rr(shape[1:], lo, hi) for _ in range(shape[0])]
    x = rr((b, l, h, p))
    dt = rr((b, l, h), 0.05, 0.6)
    A = [-rnd.uniform(0.05, 0.65) for _ in range(h)]
    Bm = rr((b, l, g, n))
    Cm = rr((b, l, g, n))
    prev = rr((b, l // chunk, h, p, n), -0.5, 0.5)
    D1 = [rnd.uniform(-1.0, 1.0) for _ in range(h)]
    z = rr((b, l, h, p), -2.0, 2.0)
    # 官方 chunk 布局的 dt
    dt4 = [[[[dt[bi][ci * chunk + li][hi] for li in range(chunk)]
             for ci in range(l // chunk)] for hi in range(h)] for bi in range(b)]
    # 官方 chunk 布局的 dA_cumsum = cumsum(dt * A, dim=-1)（chunk 内累积，不跨 chunk）
    dAcs = [[[_cumsum([dt4[bi][hi][ci][li] * A[hi] for li in range(chunk)])
              for ci in range(l // chunk)] for hi in range(h)] for bi in range(b)]
    return {"x": x, "dt": dt, "dt4": dt4, "dA_cumsum": dAcs, "A": A,
            "Bm": Bm, "Cm": Cm, "prev": prev, "D1": D1, "z": z,
            "b": b, "l": l, "h": h, "p": p, "g": g, "n": n, "chunk": chunk}


# =============================================================== check 1
def check_chunk_state(solve):
    cases = [
        _mk_case(1, 4, 1, 1, 1, 2, 2, seed=1),
        _mk_case(1, 4, 2, 3, 1, 4, 4, seed=2),
        _mk_case(2, 8, 3, 2, 3, 4, 4, seed=3),
        _mk_case(1, 6, 4, 2, 2, 4, 3, seed=4),
        _mk_case(1, 12, 2, 4, 2, 8, 6, seed=5),
        _mk_case(3, 8, 2, 2, 1, 2, 2, seed=6),
    ]
    for cs in cases:
        res = solve("chunk_state", x=cs["x"], dt=cs["dt4"], A=cs["A"],
                    B_mat=cs["Bm"], chunk_size=cs["chunk"])
        ora = _oracle_chunk_state(cs["x"], cs["dt4"], cs["A"], cs["Bm"], cs["chunk"])

        assert _close(_get(res, "dA_cumsum"), ora["dA_cumsum"]), \
            "dA_cumsum must equal the oracle (chunk-local cumsum of dt * A)"
        assert _close(_get(res, "decay_states"), ora["decay_states"]), \
            "decay_states must equal the oracle"
        assert _close(_get(res, "states"), ora["states"]), \
            "states must equal the oracle"
        assert _close(_get(res, "expanded_B"), ora["expanded_B"]), \
            "expanded_B must equal the oracle (group -> head expansion)"

        # ★ 双向断言 1：decay_states 必须是「相对 chunk 末尾」的衰减，
        #   而不是「相对 chunk 开头」的累积衰减
        dc = _get(res, "decay_states")
        dc_alt = []
        for bi in range(cs["b"]):
            ph = []
            for hi in range(cs["h"]):
                pc = []
                for ci in range(cs["l"] // cs["chunk"]):
                    row = []
                    for li in range(cs["chunk"]):
                        row.append(math.exp(cs["dt4"][bi][hi][ci][li] * cs["A"][hi]))
                    pc.append(row)
                ph.append(pc)
            dc_alt.append(ph)
        _must_differ(dc, dc_alt,
                     "decay_states must be exp(dA_cumsum[-1] - dA_cumsum[l]), "
                     "i.e. the decay measured from the END of the chunk")

        # ★ 双向断言 2：decay_states 的最后一项恒为 1.0（exp(0)）
        for bi in range(cs["b"]):
            for hi in range(cs["h"]):
                for ci in range(cs["l"] // cs["chunk"]):
                    last = dc[bi][hi][ci][cs["chunk"] - 1]
                    assert abs(last - 1.0) < 1e-12, \
                        "the last entry of decay_states must be exactly 1.0"

        # ★ 双向断言 3：expanded_B 的 group->head 映射必须是
        #   head 索引 = g * (nheads // ngroups) + hi（g 在前、h 在后）
        eb = _get(res, "expanded_B")
        hh = cs["h"] // cs["g"]
        for bi in range(cs["b"]):
            for li in range(cs["l"]):
                for gi in range(cs["g"]):
                    for hi in range(hh):
                        head = gi * hh + hi
                        assert _close(eb[bi][li][head], cs["Bm"][bi][li][gi]), \
                            "group -> head expansion must map head gi*hh+hi to group gi"

    # ---- ★ 双向断言（group -> head 展开顺序）：必须用多 group 的数据，
    #      因为 ngroups=1 时 (g h) 与 (h g) 两种顺序完全相同，断言会误报。
    mcs = _mk_case(1, 8, 4, 3, 2, 4, 4, seed=99)
    mres = solve("chunk_state", x=mcs["x"], dt=mcs["dt4"], A=mcs["A"],
                 B_mat=mcs["Bm"], chunk_size=mcs["chunk"])
    eb = _get(mres, "expanded_B")
    mhh = mcs["h"] // mcs["g"]
    eb_alt = []
    for bi in range(mcs["b"]):
        rows = []
        for li in range(mcs["l"]):
            per_head = []
            for hi in range(mhh):
                for gi in range(mcs["g"]):
                    per_head.append(mcs["Bm"][bi][li][gi])
            rows.append(per_head)
        eb_alt.append(rows)
    _must_differ(eb, eb_alt,
                 "group -> head expansion must be (g h) order, i.e. group major")
    for bi in range(mcs["b"]):
        for li in range(mcs["l"]):
            for gi in range(mcs["g"]):
                for hi in range(mhh):
                    head = gi * mhh + hi
                    assert _close(eb[bi][li][head], mcs["Bm"][bi][li][gi]), \
                        "group -> head expansion must map head gi*hh+hi to group gi"

    # ★ 双向断言：states 的 einsum 只对 chunk 内位置 l 求和，
    #   B 与 x 必须用【同一个】输出 head（不是对 head 再求一次和）
    st = _get(mres, "states")
    mdec = _get(mres, "decay_states")
    st_wrong = []
    for bi in range(mcs["b"]):
        pc = []
        for ci in range(mcs["l"] // mcs["chunk"]):
            ph = []
            for hi in range(mcs["h"]):
                pp = []
                for pi in range(mcs["p"]):
                    pn = []
                    for ni in range(mcs["n"]):
                        s = 0.0
                        for li in range(mcs["chunk"]):
                            for hj in range(mcs["h"]):
                                s += (mcs["Bm"][bi][ci * mcs["chunk"] + li][hj // mhh][ni]
                                      * mdec[bi][hi][ci][li]
                                      * mcs["dt4"][bi][hi][ci][li]
                                      * mcs["x"][bi][ci * mcs["chunk"] + li][hj][pi])
                        pn.append(s)
                    pp.append(pn)
                ph.append(pp)
            pc.append(ph)
        st_wrong.append(pc)
    _must_differ(st, st_wrong,
                 "states must sum over the chunk position only; B and x must "
                 "use the same output head (no extra sum over heads)")

    # ---- 域约束 ----
    # ★ 每个用例都必须【真正命中】目标检查：
    #   x 的形状是 [b][l][h][p]（不是 [b][h][p]），所以要多 head 就得写
    #   x = [[[[1.0],[1.0],[1.0]]]]（b=1,l=1,h=3,p=1）。
    for bad, why in (
        # chunk_size 非正 —— x/dt/B_mat 都是 (b=1,l=1,h=1,p=1) 的合法形状
        (dict(x=[[[[1.0]]]], dt=[[[[0.1]]]], A=[-0.1], B_mat=[[[[1.0]]]], chunk_size=0),
         "chunk_size must be a positive integer"),
        # nheads=3 且 ngroups=1 ⇒ 3 % 1 == 0，其实整除；要触发必须 ngroups=2
        (dict(x=[[[[1.0], [1.0], [1.0]]]],
              dt=[[[[0.1]], [[0.1]], [[0.1]]]], A=[-0.1, -0.1, -0.1],
              B_mat=[[[[1.0], [1.0]]]], chunk_size=1),
         "nheads must be divisible by ngroups"),
        # seqlen=3 不是 chunk_size=2 的整数倍；dt 给成【合法形状】(b,h,c,lc)=(1,1,1,2)，
        # 这样「删掉整除检查」之后不会再被 dt 的形状校验顶替报错
        (dict(x=[[[[1.0]], [[1.0]], [[1.0]]]],
              dt=[[[[0.1, 0.2]]]], A=[-0.1],
              B_mat=[[[[1.0]], [[1.0]], [[1.0]]]], chunk_size=2),
         "seqlen must be divisible by chunk_size"),
        # dt 布局错：官方口径是 [b][h][nchunks][chunk_size]
        (dict(x=[[[[1.0]]]], dt=[[[0.1]]], A=[-0.1],
              B_mat=[[[[1.0]]]], chunk_size=1),
         "dt must be [b][h][nchunks][chunk_size]"),
    ):
        try:
            solve("chunk_state", **bad)
        except ValueError as exc:
            # ★ 错误消息也是可观测差异：必须明确说到 chunk_size / seqlen，
            #   否则「删掉检查、让下游形状校验代为报错」也能蒙混过关。
            msg = str(exc)
            assert ("seqlen" in msg or "chunk_size" in msg or "ngroups" in msg
                    or "nheads" in msg or "dt" in msg), \
                "chunk_state must reject with a message naming the violated " \
                "constraint, got: %r" % (msg,)
        else:
            raise AssertionError("chunk_state must reject: %s" % why)




# =============================================================== check 2
def check_state_passing(solve):
    cases = [
        _mk_case(1, 4, 1, 1, 1, 2, 2, seed=11),
        _mk_case(1, 4, 2, 3, 1, 4, 4, seed=12),
        _mk_case(2, 8, 3, 2, 3, 4, 4, seed=13),
        _mk_case(1, 12, 2, 4, 2, 8, 6, seed=14),
        _mk_case(3, 8, 2, 2, 1, 2, 2, seed=15),
    ]
    for cs in cases:
        c = cs["l"] // cs["chunk"]
        h = cs["h"]
        d = cs["p"] * cs["n"]
        # 造一个 states（展平后的 headdim*dstate）
        states = []
        for bi in range(cs["b"]):
            pc = []
            for ci in range(c):
                ph = []
                for hi in range(h):
                    row = []
                    for di in range(d):
                        row.append(0.1 * ((bi + 1) * (ci + 2) * (hi + 3) * (di + 1) % 7) - 0.3)
                    ph.append(row)
                pc.append(ph)
            states.append(pc)

        # 官方 dA_chunk_cumsum = 每个 chunk 末的 dA_cumsum
        dacc_chunk = []
        for bi in range(cs["b"]):
            ph = []
            for hi in range(h):
                ph.append([cs["dt4"][bi][hi][ci][cs["chunk"] - 1] * cs["A"][hi]
                           for ci in range(c)])
            dacc_chunk.append(ph)

        res = solve("state_passing", states=states, dA_chunk_cumsum=dacc_chunk)
        ora = _oracle_state_passing(states, dacc_chunk)

        assert _close(_get(res, "decay_chunk"), ora["decay_chunk"]), \
            "decay_chunk must equal the oracle"
        assert _close(_get(res, "passed"), ora["passed"]), \
            "passed must equal the oracle"
        assert _close(_get(res, "final_state"), ora["final_state"]), \
            "final_state must equal the oracle"

        # ★ 双向断言 1：decay_chunk 必须是下三角（含对角），且尺寸是 (nchunks+1)
        dc = _get(res, "decay_chunk")
        for bi in range(cs["b"]):
            for hi in range(h):
                for zi in range(c + 1):
                    for zj in range(c + 1):
                        v = dc[bi][hi][zi][zj]
                        if zj > zi:
                            assert abs(v) < 1e-15, \
                                "decay_chunk must be lower-triangular (upper part zero)"
                        else:
                            assert v >= 0.0, "decay_chunk must be non-negative"

        # ★ 双向断言 2：decay_chunk 的对角线恒为 1.0
        for bi in range(cs["b"]):
            for hi in range(h):
                for zi in range(c + 1):
                    assert abs(dc[bi][hi][zi][zi] - 1.0) < 1e-12, \
                        "the diagonal of decay_chunk must be exactly 1.0"

        # ★ 双向断言 3：矩阵方向不能反（官方是行固定 z、列遍历 c）
        dc_alt = []
        for bi in range(cs["b"]):
            ph = []
            for hi in range(h):
                pz = []
                for zi in range(c + 1):
                    row = []
                    for zj in range(c + 1):
                        row.append(dc[bi][hi][zj][zi])
                    pz.append(row)
                ph.append(pz)
            dc_alt.append(ph)
        _must_differ(dc, dc_alt,
                     "decay_chunk must not be transposed: row is the output chunk "
                     "index, column is the summed chunk index")

        # ★ 双向断言 4：passed 的长度必须是 nchunks（不是 nchunks+1）
        ps = _get(res, "passed")
        assert len(ps) == cs["b"] and len(ps[0]) == c, \
            "passed must have exactly nchunks entries (the last one is final_state)"

        # ★ 双向断言 5：initial_states 非零时必须影响结果（不能被忽略）
        init = []
        for bi in range(cs["b"]):
            ph = []
            for hi in range(h):
                ph.append([0.25 * (di + 1) for di in range(d)])
            init.append(ph)
        res2 = solve("state_passing", states=states, dA_chunk_cumsum=dacc_chunk,
                      initial_states=init)
        ora2 = _oracle_state_passing(states, dacc_chunk)
        # oracle 用的是零初始态，这里手工算带初始态的口径
        cat = [[[[0.25 * (di + 1) for di in range(d)] for _ in range(h)]] + states[bi]
               for bi in range(cs["b"])]
        dcx = _get(res2, "decay_chunk")
        exp_passed = []
        for bi in range(cs["b"]):
            pc = []
            for zi in range(c):
                ph = []
                for hi in range(h):
                    row = []
                    for di in range(d):
                        s = 0.0
                        for zc in range(c + 1):
                            s += dcx[bi][hi][zi][zc] * cat[bi][zc][hi][di]
                        row.append(s)
                    ph.append(row)
                pc.append(ph)
            exp_passed.append(pc)
        assert _close(_get(res2, "passed"), exp_passed), \
            "initial_states must actually contribute to the passed states"

    # ---- 域约束 ----
    try:
        solve("state_passing", states=[[[[1.0, 2.0]]]], dA_chunk_cumsum=[[[0.1, 0.2]]])
    except ValueError:
        pass
    else:
        raise AssertionError("state_passing must reject a mismatched dA_chunk_cumsum")


# =============================================================== check 3
def check_chunk_scan(solve):
    cases = [
        _mk_case(1, 4, 1, 1, 1, 2, 2, seed=21),
        _mk_case(1, 4, 2, 3, 1, 4, 4, seed=22),
        _mk_case(2, 8, 3, 2, 3, 4, 4, seed=23),
        _mk_case(1, 12, 2, 4, 2, 8, 6, seed=24),
        _mk_case(1, 6, 4, 2, 2, 4, 3, seed=25),
        _mk_case(3, 8, 2, 2, 1, 2, 2, seed=26),
    ]
    for cs in cases:
        c = cs["l"] // cs["chunk"]
        res = solve("chunk_scan", B_mat=cs["Bm"], C_mat=cs["Cm"], x=cs["x"],
                    dt=cs["dt4"], dA_cumsum=cs["dA_cumsum"], prev_states=cs["prev"],
                    D=cs["D1"], z=cs["z"])
        ora = _oracle_chunk_scan(cs["Bm"], cs["Cm"], cs["x"], cs["dt4"],
                                 cs["dA_cumsum"], cs["prev"], D=cs["D1"], z=cs["z"])
        assert _close(_get(res, "CB"), ora["CB"]), "CB must equal the oracle"
        assert _close(_get(res, "scores_decay"), ora["scores_decay"]), \
            "scores_decay must equal the oracle"
        assert _close(_get(res, "out"), ora["out"]), "out must equal the oracle"

        # ★ 双向断言 1：scores_decay 必须是下三角（含对角）⇒ 因果
        sd = _get(res, "scores_decay")
        for bi in range(cs["b"]):
            for ci in range(c):
                for hi in range(cs["h"]):
                    for li in range(cs["chunk"]):
                        for si in range(cs["chunk"]):
                            if si > li:
                                assert abs(sd[bi][ci][hi][li][si]) < 1e-15, \
                                    "scores_decay must be causal (lower triangular)"

        # ★ 双向断言 2：decay 的方向不能反
        dec = _get(res, "decay")
        for bi in range(cs["b"]):
            for hi in range(cs["h"]):
                for ci in range(c):
                    for li in range(cs["chunk"]):
                        for si in range(cs["chunk"]):
                            want = math.exp(cs["dA_cumsum"][bi][hi][ci][li]
                                             - cs["dA_cumsum"][bi][hi][ci][si])
                            assert abs(dec[bi][hi][ci][li][si] - want) < 1e-9, \
                                "decay[l,s] must be exp(dA_cumsum[l] - dA_cumsum[s])"

        # ★ 双向断言 3：z 是乘性门控（silu），不是加性
        out_z = _get(res, "out")
        res_nz = solve("chunk_scan", B_mat=cs["Bm"], C_mat=cs["Cm"], x=cs["x"],
                       dt=cs["dt4"], dA_cumsum=cs["dA_cumsum"], prev_states=cs["prev"],
                       D=cs["D1"], z=None)
        base = _get(res_nz, "out")
        add_z = []
        for bi in range(cs["b"]):
            rows = []
            for li in range(cs["l"]):
                ph = []
                for hi in range(cs["h"]):
                    row = []
                    for pi in range(cs["p"]):
                        row.append(base[bi][li][hi][pi] + cs["z"][bi][li][hi][pi])
                    ph.append(row)
                rows.append(ph)
            add_z.append(rows)
        _must_differ(out_z, add_z, "z must be a multiplicative silu gate, not additive")

    # ---- 域约束 ----
    for bad, why in (
        (dict(B_mat=[[[[1.0]], [[1.0]]]], C_mat=[[[[1.0]], [[1.0]]]],
              x=[[[[1.0]], [[1.0]]]], dt=[[[[0.1], [0.1]]]],
              dA_cumsum=[[[[-0.1, -0.2]]]], prev_states=[[[[[1.0]]]]]),
         "seqlen must equal nchunks * chunk_size"),
        (dict(B_mat=[[[[1.0]]]], C_mat=[[[[1.0]]]], x=[[[[1.0]]]],
              dt=[[[[0.1]]]], dA_cumsum=[[[[-0.1]]]],
              prev_states=[[[[[1.0]]]]], D=[1.0, 2.0, 3.0]),
         "D must be [nheads] or [nheads][headdim]"),
    ):
        try:
            solve("chunk_scan", **bad)
        except ValueError:
            pass
        else:
            raise AssertionError("chunk_scan must reject: %s" % why)


# =============================================================== check 4
def check_ssd_forward(solve):
    cases = [
        _mk_case(1, 4, 1, 1, 1, 2, 2, seed=31),
        _mk_case(1, 4, 2, 3, 1, 4, 4, seed=32),
        _mk_case(2, 8, 3, 2, 3, 4, 4, seed=33),
        _mk_case(1, 12, 2, 4, 2, 8, 6, seed=34),
        _mk_case(3, 8, 2, 2, 1, 2, 2, seed=35),
        _mk_case(1, 16, 2, 5, 1, 6, 8, seed=36),
    ]
    for cs in cases:
        c = cs["l"] // cs["chunk"]
        res = solve("ssd_forward", x=cs["x"], dt=cs["dt"], A=cs["A"],
                    B_mat=cs["Bm"], C_mat=cs["Cm"], chunk_size=cs["chunk"],
                    D=cs["D1"], z=cs["z"])
        # oracle：三段手工串起来
        cs_state = _oracle_chunk_state(cs["x"], cs["dt4"], cs["A"], cs["Bm"], cs["chunk"])
        flat = []
        for bi in range(cs["b"]):
            pc = []
            for ci in range(c):
                ph = []
                for hi in range(cs["h"]):
                    row = []
                    for pi in range(cs["p"]):
                        for ni in range(cs["n"]):
                            row.append(cs_state["states"][bi][ci][hi][pi][ni])
                    ph.append(row)
                pc.append(ph)
            flat.append(pc)
        dacc_chunk = []
        for bi in range(cs["b"]):
            ph = []
            for hi in range(cs["h"]):
                ph.append([cs_state["dA_cumsum"][bi][hi][ci][cs["chunk"] - 1]
                           for ci in range(c)])
            dacc_chunk.append(ph)
        sp = _oracle_state_passing(flat, dacc_chunk)
        prev = []
        for bi in range(cs["b"]):
            pc = []
            for ci in range(c):
                ph = []
                for hi in range(cs["h"]):
                    pp = []
                    for pi in range(cs["p"]):
                        pn = []
                        for ni in range(cs["n"]):
                            pn.append(sp["passed"][bi][ci][hi][pi * cs["n"] + ni])
                        pp.append(pn)
                    ph.append(pp)
                pc.append(ph)
            prev.append(pc)
        scan = _oracle_chunk_scan(cs["Bm"], cs["Cm"], cs["x"], cs["dt4"],
                                   cs_state["dA_cumsum"], prev, D=cs["D1"], z=cs["z"])
        assert _close(_get(res, "out"), scan["out"]), \
            "ssd_forward out must equal the oracle (three-stage pipeline)"
        assert _close(_get(res, "dA_cumsum"), cs_state["dA_cumsum"]), \
            "ssd_forward dA_cumsum must equal the oracle"
        assert _close(_get(res, "states"), cs_state["states"]), \
            "ssd_forward states must equal the oracle"
        assert _close(_get(res, "prev_states"), prev), \
            "ssd_forward prev_states must equal the oracle (state passing result)"
        assert _close(_get(res, "final_state"), sp["final_state"]), \
            "ssd_forward final_state must equal the oracle"

        # ★ 双向断言 1：dt_bias 必须在 softplus **之前**相加
        bias = [0.1 * (i + 1) for i in range(cs["h"])]
        r_bias = solve("ssd_forward", x=cs["x"], dt=cs["dt"], A=cs["A"],
                        B_mat=cs["Bm"], C_mat=cs["Cm"], chunk_size=cs["chunk"],
                        D=cs["D1"], z=cs["z"], dt_bias=bias)
        r_bias_soft = solve("ssd_forward", x=cs["x"], dt=cs["dt"], A=cs["A"],
                             B_mat=cs["Bm"], C_mat=cs["Cm"], chunk_size=cs["chunk"],
                             D=cs["D1"], z=cs["z"], dt_bias=bias, dt_softplus=True)
        r_soft = solve("ssd_forward", x=cs["x"], dt=cs["dt"], A=cs["A"],
                        B_mat=cs["Bm"], C_mat=cs["Cm"], chunk_size=cs["chunk"],
                        D=cs["D1"], z=cs["z"], dt_softplus=True)
        _must_differ(_get(r_bias, "dA_cumsum"), _get(r_bias_soft, "dA_cumsum"),
                     "dt_bias must be added BEFORE softplus, not after")
        _must_differ(_get(r_bias, "dA_cumsum"), _get(r_soft, "dA_cumsum"),
                     "dt_bias must change the result")

        # ★ 双向断言 2：dt_bias 的作用是【加性】且逐 head 独立
        db = _get(r_bias, "dA_cumsum")
        d0 = _get(res, "dA_cumsum")
        for bi in range(cs["b"]):
            for hi in range(cs["h"]):
                for ci in range(c):
                    for li in range(cs["chunk"]):
                        # dt_bias 先加到 dt、再乘 A，最后 cumsum ⇒ 增量是
                        # (li+1) * bias[hi] * A[hi]（累积效应）
                        want = (d0[bi][hi][ci][li]
                                + (li + 1) * bias[hi] * cs["A"][hi])
                        assert abs(db[bi][hi][ci][li] - want) < 1e-9, \
                            "dt_bias must be added to dt BEFORE the A scaling"

    # ---- 域约束：seqlen 非 chunk 整数倍 → 官方会崩，本题要求显式报错 ----
    try:
        solve("ssd_forward", x=[[[[1.0]]], [[[1.0]]], [[[1.0]]]],
              dt=[[[0.1]], [[0.1]], [[0.1]]], A=[-0.1],
              B_mat=[[[[1.0]]], [[[1.0]]], [[[1.0]]]],
              C_mat=[[[[1.0]]], [[[1.0]]], [[[1.0]]]],
              chunk_size=2)
    except ValueError as exc:
        # ★ 必须由 ssd_forward 自己报这个错（消息里带 ssd_forward），
        #   而不是让下游的 chunk_state 代为报错 —— 否则「删掉 ssd_forward 的
        #   整除检查」这个变异就能蒙混过关。
        assert "ssd_forward" in str(exc), \
            "ssd_forward must itself reject a non-multiple seqlen (the official " \
            "reference crashes there), got: %r" % (str(exc),)
    else:
        raise AssertionError("ssd_forward must reject a seqlen that is not a "
                             "multiple of chunk_size (the official reference crashes there)")
