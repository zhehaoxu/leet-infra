"""order 844 判题用例（暂存）：RetNet 保留机制与 xPos 相对位置的账。

独立 oracle（不 import solution），每个 check_* 含手算锚点 + 网格对拍。

口径：
  - mode 1 / mode 2 全链路精确有理数（fractions.Fraction）比较，逐位相等。
  - mode 3 / mode 4 含无理幂与三角函数，按容差判定：
      mode 3：相对容差 1e-9
      mode 4：绝对容差 1e-8（xPos 分支本质是浮点近似；语义错误会差 1e-3 以上）
  - retention 的三种表示（parallel / recurrent / chunkwise）在纯骨架下严格等价。
"""

import math
from fractions import Fraction

_XPOS_SCALE_BASE = 512
_GN_EPS = 1e-5


# ------------------------------------------------------------------ 工具
def _f(x):
    if isinstance(x, Fraction):
        return x
    if isinstance(x, bool):
        return Fraction(int(x))
    if isinstance(x, int):
        return Fraction(x)
    if isinstance(x, str):
        return Fraction(x)
    return Fraction(str(float(x)))


def _f32(x):
    import struct
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


def _cmp(actual, expected, label):
    """逐位相等比较（Fraction 或 int）。"""
    a = _f(actual)
    e = _f(expected)
    assert a == e, "%s: got %s expected %s" % (label, a, e)


def _cmp_close(actual, expected, label, rel_tol=1e-9, abs_tol=0.0):
    a = float(actual)
    e = float(expected)
    err = abs(a - e)
    scale = max(abs(a), abs(e))
    ok = err <= abs_tol or (scale > 0 and err / scale <= rel_tol)
    assert ok, "%s: got %.17g expected %.17g (abs err %.3e)" % (label, a, e, err)


def _cmp_matrix(actual, expected, label, rel_tol=1e-9, abs_tol=0.0):
    assert len(actual) == len(expected), "%s: rows %d != %d" % (label, len(actual), len(expected))
    for i, (ra, re) in enumerate(zip(actual, expected)):
        assert len(ra) == len(re), "%s: row %d width %d != %d" % (label, i, len(ra), len(re))
        for j, (va, ve) in enumerate(zip(ra, re)):
            _cmp_close(va, ve, "%s[%d][%d]" % (label, i, j), rel_tol, abs_tol)


def _cmp_deep(actual, expected, label):
    """嵌套 list 的逐位相等比较。"""
    if isinstance(expected, list):
        assert isinstance(actual, list), "%s: expected list, got %r" % (label, type(actual))
        assert len(actual) == len(expected), "%s: len %d != %d" % (label, len(actual), len(expected))
        for i, (a, e) in enumerate(zip(actual, expected)):
            _cmp_deep(a, e, "%s[%d]" % (label, i))
    else:
        _cmp(actual, expected, label)


class _Rng:
    """确定性 LCG，保证用例可复现。"""

    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFFFFFFFFFF

    def _next(self):
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return (self.state >> 33) & 0x7FFFFFFF

    def randint(self, lo, hi):
        """闭区间 [lo, hi]。"""
        return lo + self._next() % (hi - lo + 1)

    def choice(self, seq):
        return seq[self._next() % len(seq)]


def _rand_matrix(rng, rows, cols, lo=-6, hi=6):
    return [[rng.randint(lo, hi) for _ in range(cols)] for _ in range(rows)]


def _rand_heads(rng, heads, seq_len, width, lo=-6, hi=6):
    return [_rand_matrix(rng, seq_len, width, lo, hi) for _ in range(heads)]


# ------------------------------------------------------------------ 独立 oracle
def oracle_decay_matrix(gamma, seq_len):
    """D[n][m] = gamma ** (n - m) 当 n >= m，否则 0。"""
    g = _f(gamma)
    powers = []
    cur = Fraction(1)
    for _ in range(seq_len):
        powers.append(cur)
        cur = cur * g
    D = []
    for n in range(seq_len):
        row = []
        for m in range(seq_len):
            row.append(powers[n - m] if n >= m else Fraction(0))
        D.append(row)
    return powers, D


def oracle_parallel(Q, K, V, gamma):
    """scores[h][n][m] = D[n][m] * (q_h[n] · k_h[m])；out[n][h] = sum_m scores * V。"""
    seq_len = len(Q[0])
    _powers, D = oracle_decay_matrix(gamma, seq_len)
    heads = len(Q)
    width = len(Q[0][0])
    vdim = len(V[0][0])
    scores = []
    for h in range(heads):
        hs = []
        for n in range(seq_len):
            row = []
            for m in range(seq_len):
                qk = Fraction(0)
                for d in range(width):
                    qk += _f(Q[h][n][d]) * _f(K[h][m][d])
                row.append(qk * D[n][m])
            hs.append(row)
        scores.append(hs)
    out = []
    for n in range(seq_len):
        orow = []
        for h in range(heads):
            acc = [Fraction(0)] * vdim
            for m in range(seq_len):
                s = scores[h][n][m]
                if s == 0:
                    continue
                for e in range(vdim):
                    acc[e] += s * _f(V[h][m][e])
            orow.append(acc)
        out.append(orow)
    return scores, out


def oracle_recurrent(Q, K, V, gamma):
    """s_n = gamma * s_{n-1} + K_n^T V_n（K_n/V_n 只有当前一步）；out_n = Q_n @ s_n。"""
    g = _f(gamma)
    seq_len = len(Q[0])
    heads = len(Q)
    width = len(Q[0][0])
    vdim = len(V[0][0])
    cur = [[[Fraction(0)] * vdim for _ in range(width)] for _ in range(heads)]
    states = []
    outs = []
    for n in range(seq_len):
        new = []
        for h in range(heads):
            s = []
            for d in range(width):
                row = []
                for e in range(vdim):
                    acc = g * cur[h][d][e] + _f(K[h][n][d]) * _f(V[h][n][e])
                    row.append(acc)
                s.append(row)
            new.append(s)
        cur = new
        states.append([list(r) for r in new])
        orow = []
        for h in range(heads):
            acc = [Fraction(0)] * vdim
            for d in range(width):
                qv = _f(Q[h][n][d])
                if qv == 0:
                    continue
                for e in range(vdim):
                    acc[e] += qv * cur[h][d][e]
            orow.append(acc)
        outs.append(orow)
    return states, outs


def oracle_chunkwise(Q, K, V, gamma, chunk_size):
    """分块表示：inner + cross（cross 用【旧状态】r_i_1）。"""
    g = _f(gamma)
    seq_len = len(Q[0])
    heads = len(Q)
    width = len(Q[0][0])
    vdim = len(V[0][0])
    r_prev = [[[Fraction(0)] * vdim for _ in range(width)] for _ in range(heads)]
    outs = []
    block_states = []
    nblocks = (seq_len + chunk_size - 1) // chunk_size
    for bi in range(nblocks):
        lo = bi * chunk_size
        hi = min(lo + chunk_size, seq_len)
        c = hi - lo
        _powers, Dc = oracle_decay_matrix(gamma, c)
        d_last = Dc[-1]
        old_r = r_prev
        new_r = []
        for h in range(heads):
            s = []
            for d in range(width):
                row = []
                for e in range(vdim):
                    # 官方 forward_chunkwise 的 chunk_size = x_i.shape[1]（本块实际长度）
                    acc = (g ** c) * old_r[h][d][e]
                    for m in range(c):
                        acc += _f(K[h][lo + m][d]) * (_f(V[h][lo + m][e]) * d_last[m])
                    row.append(acc)
                s.append(row)
            new_r.append(s)
        # inner
        blk_scores = []
        for h in range(heads):
            hs = []
            for j in range(c):
                row = []
                for m in range(c):
                    qk = Fraction(0)
                    for d in range(width):
                        qk += _f(Q[h][lo + j][d]) * _f(K[h][lo + m][d])
                    row.append(qk * Dc[j][m])
                hs.append(row)
            blk_scores.append(hs)
        # cross 的衰减因子 e[j] = gamma ** (j + 1)
        e = [g ** (j + 1) for j in range(c)]
        for j in range(c):
            orow = []
            for h in range(heads):
                acc = [Fraction(0)] * vdim
                for m in range(c):
                    s = blk_scores[h][j][m]
                    if s == 0:
                        continue
                    for d in range(vdim):
                        acc[d] += s * _f(V[h][lo + m][d])
                for d in range(width):
                    qv = _f(Q[h][lo + j][d])
                    if qv == 0:
                        continue
                    for eidx in range(vdim):
                        acc[eidx] += qv * old_r[h][d][eidx] * e[j]
                orow.append(acc)
            outs.append(orow)
        r_prev = new_r
        block_states.append([[list(row) for row in head] for head in new_r])
    return outs, block_states


def oracle_xpos_base(head_dim):
    """base[j] = f32(f32(2j + 0.4*hd) / f32(1.4*hd))（官方 scale buffer 是 float32）。"""
    den = _f32(1.4 * head_dim)
    out = []
    for j in range(head_dim // 2):
        num = _f32(2 * j + 0.4 * head_dim)
        out.append(_f32(num / den))
    return out


def oracle_xpos_scale(head_dim, seq_len, offset):
    """scale[pos][j] = base[j] ** (pos/512)；sin/cos 由 fixed_pos_embedding 得到。"""
    base = oracle_xpos_base(head_dim)
    dim = len(base)
    total = seq_len + offset
    inv_freq = []
    for j in range(dim):
        ratio = _f32(float(j) / dim)
        inv_freq.append(_f32(1.0 / _f32(10000.0 ** ratio)))
    scale = []
    sin = []
    cos = []
    for pos in range(total):
        srow = []
        sinrow = []
        cosrow = []
        for j in range(dim):
            srow.append(float(base[j]) ** _f32(float(pos) / _XPOS_SCALE_BASE))
            ang = _f32(_f32(float(pos)) * inv_freq[j])
            sinrow.append(math.sin(ang))
            cosrow.append(math.cos(ang))
        scale.append(srow)
        sin.append(sinrow)
        cos.append(cosrow)
    if total > seq_len:
        scale = scale[-seq_len:]
        sin = sin[-seq_len:]
        cos = cos[-seq_len:]
    return base, scale, sin, cos


def oracle_ms_gammas(heads):
    """gamma_h = 1 - exp(linspace(log(1/32), log(1/512), heads)_h)。"""
    lo = math.log(1.0 / 32.0)
    hi = math.log(1.0 / 512.0)
    out = []
    for h in range(heads):
        if heads == 1:
            t = lo
        else:
            t = lo + (hi - lo) * h / (heads - 1)
        out.append(1.0 - math.exp(t))
    return out


def oracle_group_norm(y, num_groups):
    """对 (L, C) 按连续通道分 G 组做归一化：方差有偏（1/N），eps = 1e-5。"""
    L = len(y)
    C = len(y[0])
    gsize = C // num_groups
    out = []
    for n in range(L):
        row = []
        for g in range(num_groups):
            seg = y[n][g * gsize:(g + 1) * gsize]
            m = sum(seg) / gsize
            var = sum((v - m) ** 2 for v in seg) / gsize
            denom = math.sqrt(var + _GN_EPS)
            for v in seg:
                row.append((v - m) / denom)
        out.append(row)
    return out


def oracle_multiscale(X, W_Q, W_K, W_V, W_G, W_O, heads):
    """多尺度保留：逐 head retention（带 xPos）-> cat -> GroupNorm -> (swish(X@W_G)*Y)@W_O。"""
    L = len(X)
    hidden = len(X[0])
    head_size = hidden // heads
    gammas = oracle_ms_gammas(heads)

    def matmul(a, b):
        K_ = len(a[0])
        M_ = len(b[0])
        out = []
        for i in range(len(a)):
            row = []
            for j in range(M_):
                acc = 0.0
                for k in range(K_):
                    acc += a[i][k] * b[k][j]
                row.append(acc)
            out.append(row)
        return out

    base, scale, sin, cos = oracle_xpos_scale(head_size, L, 0)
    kbase, kscale, ksin, kcos = oracle_xpos_scale(head_size, L, 0)

    def dup_interleave(t, sc):
        out = []
        for i in range(len(t)):
            res = []
            for j in range(len(t[i])):
                v = t[i][j] * sc[i][j]
                res.append(v)
                res.append(v)
            out.append(res)
        return out

    def apply_xpos(x, sin_, cos_, sc):
        sin_d = dup_interleave(sin_, sc)
        cos_d = dup_interleave(cos_, sc)
        out = []
        for n in range(len(x)):
            row = x[n]
            x1 = row[0::2]
            x2 = row[1::2]
            rot = []
            for a, b in zip(x2, x1):
                rot.append(-a)
                rot.append(b)
            r = []
            for d in range(len(row)):
                r.append(row[d] * cos_d[n][d] + rot[d] * sin_d[n][d])
            out.append(r)
        return out

    ys = []
    for h in range(heads):
        q = apply_xpos(matmul(X, W_Q[h]), sin, cos, scale)
        k = apply_xpos(matmul(X, W_K[h]), ksin, kcos, [[1.0 / v for v in row] for row in kscale])
        v = matmul(X, W_V[h])
        D = [[gammas[h] ** (n - m) if n >= m else 0.0 for m in range(L)] for n in range(L)]
        out_h = []
        for n in range(L):
            row = []
            for e in range(head_size):
                acc = 0.0
                for m in range(L):
                    qk = 0.0
                    for d in range(head_size):
                        qk += q[n][d] * k[m][d]
                    acc += qk * D[n][m] * v[m][e]
                row.append(acc)
            out_h.append(row)
        ys.append(out_h)

    y_cat = []
    for n in range(L):
        row = []
        for h in range(heads):
            row.extend(ys[h][n])
        y_cat.append(row)

    y_gn = oracle_group_norm(y_cat, heads)

    xg = matmul(X, W_G)
    gated = []
    for n in range(L):
        row = []
        for d in range(len(xg[n])):
            sw = xg[n][d] * (1.0 / (1.0 + math.exp(-xg[n][d])))
            row.append(sw * y_gn[n][d])
        gated.append(row)

    out = matmul(gated, W_O)
    return gammas, y_cat, y_gn, out


# ------------------------------------------------------------------ check 1
def check_decay_matrix(solve):
    # --- 手算锚点 1：gamma = 1/2, L = 3
    r = solve("decay_matrix", gamma="1/2", seq_len=3)
    _cmp_deep(r["gamma_powers"], [Fraction(1), Fraction(1, 2), Fraction(1, 4)], "dm.a1.powers")
    _cmp_deep(r["D"], [
        [Fraction(1), Fraction(0), Fraction(0)],
        [Fraction(1, 2), Fraction(1), Fraction(0)],
        [Fraction(1, 4), Fraction(1, 2), Fraction(1)],
    ], "dm.a1.D")

    # --- 手算锚点 2：gamma = 2/3, L = 2；对角线恒为 1
    r = solve("decay_matrix", gamma="2/3", seq_len=2)
    _cmp_deep(r["D"], [
        [Fraction(1), Fraction(0)],
        [Fraction(2, 3), Fraction(1)],
    ], "dm.a2.D")

    # --- 手算锚点 3：gamma = 1/2, L = 4，powers 逐项为 2 的负幂
    r = solve("decay_matrix", gamma="1/2", seq_len=4)
    _cmp_deep(r["gamma_powers"], [Fraction(1), Fraction(1, 2), Fraction(1, 4), Fraction(1, 8)],
              "dm.a3.powers")

    # --- 网格对拍
    rng = _Rng(8441)
    for _ in range(40):
        gamma = "%d/%d" % (rng.randint(1, 20), rng.randint(21, 40))
        L = rng.randint(1, 8)
        r = solve("decay_matrix", gamma=gamma, seq_len=L)
        powers, D = oracle_decay_matrix(gamma, L)
        _cmp_deep(r["gamma_powers"], powers, "dm.grid.powers")
        _cmp_deep(r["D"], D, "dm.grid.D")

    # --- 非法输入
    for bad_gamma in ("0", "1", "-1/2", "5/4", "2"):
        try:
            solve("decay_matrix", gamma=bad_gamma, seq_len=3)
        except ValueError:
            pass
        else:
            raise AssertionError("dm: gamma %r should be rejected" % (bad_gamma,))
    for bad_len in (0, -1):
        try:
            solve("decay_matrix", gamma="1/2", seq_len=bad_len)
        except ValueError:
            pass
        else:
            raise AssertionError("dm: seq_len %r should be rejected" % (bad_len,))


# ------------------------------------------------------------------ check 2
def check_retention_forms(solve):
    # --- 手算锚点 1：单 head、L=2、gamma=1/2
    Q = [[[1, 0], [0, 1]]]
    K = [[[1, 0], [1, 0]]]
    V = [[[1, 0], [0, 1]]]
    r = solve("retention_forms", Q=Q, K=K, V=V, gamma="1/2", chunk_size=2)
    # scores[0][0][0] = 1 * (q0·k0) = 1 * 1 = 1
    _cmp(r["scores"][0][0][0], Fraction(1), "rf.a1.scores[0][0][0]")
    # scores[0][0][1] = 0（n<m 被 mask）
    _cmp(r["scores"][0][0][1], Fraction(0), "rf.a1.scores[0][0][1]")
    # scores[0][1][0] = 1/2 * (q1·k0) = 1/2 * (0*1 + 1*0) = 0
    _cmp(r["scores"][0][1][0], Fraction(0), "rf.a1.scores[0][1][0]")
    # scores[0][1][1] = 1 * (q1·k1) = 1 * (0*1+1*0)=0
    _cmp(r["scores"][0][1][1], Fraction(0), "rf.a1.scores[0][1][1]")
    # parallel[0][0] = sum_m scores[0][0][m] * V[m] = 1 * [1,0] = [1,0]
    _cmp_deep(r["parallel"][0][0], [Fraction(1), Fraction(0)], "rf.a1.par[0][0]")
    # parallel[1][0] = scores[1][0]*V[0] + scores[1][1]*V[1] = 0
    _cmp_deep(r["parallel"][1][0], [Fraction(0), Fraction(0)], "rf.a1.par[1][0]")

    # --- 手算锚点 2：单 head、L=2、gamma=1/2，构造非平凡的 scores
    Q = [[[1, 2], [3, 4]]]
    K = [[[1, 0], [0, 1]]]
    V = [[[1, 0], [1, 1]]]
    r = solve("retention_forms", Q=Q, K=K, V=V, gamma="1/2", chunk_size=2)
    # scores[0][0][0] = 1 * (1*1 + 2*0) = 1
    _cmp(r["scores"][0][0][0], Fraction(1), "rf.a2.scores[0][0][0]")
    # scores[0][1][0] = 1/2 * (3*1 + 4*0) = 3/2
    _cmp(r["scores"][0][1][0], Fraction(3, 2), "rf.a2.scores[0][1][0]")
    # scores[0][1][1] = 1 * (3*0 + 4*1) = 4
    _cmp(r["scores"][0][1][1], Fraction(4), "rf.a2.scores[0][1][1]")
    # parallel[1][0] = (3/2)*V[0] + 4*V[1] = (3/2)*[1,0] + 4*[1,1] = [11/2, 4]
    _cmp_deep(r["parallel"][1][0], [Fraction(11, 2), Fraction(4)], "rf.a2.par[1][0]")

    # --- 手算锚点 3：recurrent 的状态递推（单 head、width=2、vdim=2）
    # s_0 = 0*0 + K_0^T V_0（单步外积）
    r = solve("retention_forms", Q=Q, K=K, V=V, gamma="1/2", chunk_size=2)
    # states[0][0][d][e] = K_0^T V_0 的单步外积：K_0 = [1,0]，V_0 = [1,0]
    _cmp_deep(r["states"][0][0], [[Fraction(1), Fraction(0)], [Fraction(0), Fraction(0)]],
              "rf.a3.states[0][0]")
    # s_1 = (1/2)*s_0 + K_1^T V_1，K_1 = [0,1]，V_1 = [1,1]
    #   -> 第 0 行 = (1/2)*[1,0] + [0*1, 0*1] = [1/2, 0]
    #   -> 第 1 行 = (1/2)*[0,0] + [1*1, 1*1] = [1, 1]
    _cmp_deep(r["states"][1][0], [
        [Fraction(1, 2), Fraction(0)],
        [Fraction(1), Fraction(1)],
    ], "rf.a3.states[1][0]")

    # --- 网格对拍 + 三表示等价
    rng = _Rng(8442)
    for _ in range(50):
        heads = rng.randint(1, 3)
        L = rng.randint(1, 7)
        width = rng.randint(1, 3)
        vdim = rng.randint(1, 3)
        gamma = "%d/%d" % (rng.randint(1, 15), rng.randint(16, 30))
        Q = _rand_heads(rng, heads, L, width)
        K = _rand_heads(rng, heads, L, width)
        V = _rand_heads(rng, heads, L, vdim)
        chunk = rng.randint(1, L + 2)
        r = solve("retention_forms", Q=Q, K=K, V=V, gamma=gamma, chunk_size=chunk)
        scores, par = oracle_parallel(Q, K, V, gamma)
        states, rec = oracle_recurrent(Q, K, V, gamma)
        chw, bstates = oracle_chunkwise(Q, K, V, gamma, chunk)
        _cmp_deep(r["scores"], scores, "rf.grid.scores")
        _cmp_deep(r["parallel"], par, "rf.grid.parallel")
        _cmp_deep(r["recurrent"], rec, "rf.grid.recurrent")
        _cmp_deep(r["chunkwise"], chw, "rf.grid.chunkwise")
        _cmp_deep(r["states"], states, "rf.grid.states")
        _cmp_deep(r["block_states"], bstates, "rf.grid.block_states")
        # ★ 三表示必须严格等价（纯骨架下无 xPos 末块错位问题）
        _cmp_deep(r["parallel"], r["recurrent"], "rf.grid.par_eq_rec")
        _cmp_deep(r["parallel"], r["chunkwise"], "rf.grid.par_eq_chw")

    # --- 非法输入
    try:
        solve("retention_forms", Q=[], K=[], V=[], gamma="1/2", chunk_size=2)
    except ValueError:
        pass
    else:
        raise AssertionError("rf: empty Q should be rejected")
    try:
        solve("retention_forms", Q=[[[1, 2]]], K=[[[1, 2]]], V=[[[1, 2]]], gamma="1/2", chunk_size=0)
    except ValueError:
        pass
    else:
        raise AssertionError("rf: chunk_size 0 should be rejected")
    try:
        solve("retention_forms", Q=[[[1, 2]]], K=[[[1, 2], [1, 1]]], V=[[[1, 2]]], gamma="1/2", chunk_size=1)
    except ValueError:
        pass
    else:
        raise AssertionError("rf: mismatched K length should be rejected")
    # K 的 head 数与 Q 不一致
    try:
        solve("retention_forms", Q=[[[1, 2]]], K=[[[1, 2]], [[3, 4]]], V=[[[1, 2]]], gamma="1/2", chunk_size=1)
    except ValueError:
        pass
    else:
        raise AssertionError("rf: mismatched K head count should be rejected")
    # V 的 head 数与 Q 不一致
    try:
        solve("retention_forms", Q=[[[1, 2]]], K=[[[1, 2]]], V=[[[1, 2]], [[3, 4]]], gamma="1/2", chunk_size=1)
    except ValueError:
        pass
    else:
        raise AssertionError("rf: mismatched V head count should be rejected")
    # V 的行宽为空（[[[]]]）
    try:
        solve("retention_forms", Q=[[[1, 2]]], K=[[[1, 2]]], V=[[[]]], gamma="1/2", chunk_size=1)
    except ValueError:
        pass
    else:
        raise AssertionError("rf: empty V row width should be rejected")
    # gamma 越界
    for bad_gamma in ("0", "1", "-1/2", "5/4"):
        try:
            solve("retention_forms", Q=[[[1, 2]]], K=[[[1, 2]]], V=[[[1, 2]]], gamma=bad_gamma, chunk_size=1)
        except ValueError:
            pass
        else:
            raise AssertionError("rf: gamma %r should be rejected" % (bad_gamma,))


# ------------------------------------------------------------------ check 3
def check_xpos_scale(solve):
    # --- 手算锚点 1：head_dim=4 的 base（float32 语义）
    r = solve("xpos_scale", head_dim=4, seq_len=3, offset=0, downscale=False)
    exp_base = oracle_xpos_base(4)
    assert len(r["base"]) == len(exp_base), "xs.a1.base width"
    for j in range(len(exp_base)):
        _cmp_close(r["base"][j], exp_base[j], "xs.a1.base[%d]" % j, rel_tol=1e-12)
    # base[0] = (0 + 0.4*4)/(1.4*4) = 1.6/5.6 = 2/7（float32 语义）
    _cmp_close(r["base"][0], float(_f32(_f32(1.6) / _f32(5.6))), "xs.a1.base0", rel_tol=1e-12)
    # scale 第 0 行全为 1（base ** 0）
    for j in range(2):
        _cmp_close(r["scale"][0][j], 1.0, "xs.a1.scale[0][%d]" % j, rel_tol=1e-12)

    # --- 手算锚点 2：sin/cos 的第 0 行（angle = 0）恒为 sin=0, cos=1
    r = solve("xpos_scale", head_dim=8, seq_len=4, offset=0, downscale=False)
    for j in range(4):
        _cmp_close(r["sin"][0][j], 0.0, "xs.a2.sin[0][%d]" % j, abs_tol=1e-15)
        _cmp_close(r["cos"][0][j], 1.0, "xs.a2.cos[0][%d]" % j, abs_tol=1e-15)

    # --- 手算锚点 3：downscale 只把 applied_scale 取倒数，sin/cos 不变
    up = solve("xpos_scale", head_dim=8, seq_len=4, offset=0, downscale=False)
    dn = solve("xpos_scale", head_dim=8, seq_len=4, offset=0, downscale=True)
    _cmp_matrix(dn["sin"], up["sin"], "xs.a3.sin_same", rel_tol=1e-15)
    _cmp_matrix(dn["cos"], up["cos"], "xs.a3.cos_same", rel_tol=1e-15)
    for i in range(4):
        for j in range(4):
            _cmp_close(dn["applied_scale"][i][j], 1.0 / up["applied_scale"][i][j],
                       "xs.a3.applied[%d][%d]" % (i, j), rel_tol=1e-12)

    # --- 手算锚点 4：offset 截断取最后 seq_len 行
    off = solve("xpos_scale", head_dim=8, seq_len=3, offset=2, downscale=False)
    base, sc, sn, cs = oracle_xpos_scale(8, 3, 2)
    _cmp_matrix(off["scale"], sc, "xs.a4.scale", rel_tol=1e-9)
    # 截断后第一行对应 pos = 2（total=5，取最后 3 行）
    full = oracle_xpos_scale(8, 5, 0)
    # total = 3 + 2 = 5 > 3 -> 取最后 3 行，第 0 行对应 pos = 2
    _cmp_close(off["scale"][0][0], full[1][2][0], "xs.a4.first_row", rel_tol=1e-12)

    # --- 网格对拍
    rng = _Rng(8443)
    for _ in range(45):
        head_dim = 2 * rng.randint(1, 6)
        L = rng.randint(1, 8)
        offset = rng.randint(0, 5)
        ds = rng.choice([False, True])
        r = solve("xpos_scale", head_dim=head_dim, seq_len=L, offset=offset, downscale=ds)
        base, sc, sn, cs = oracle_xpos_scale(head_dim, L, offset)
        _cmp_close(r["base"][0], base[0], "xs.grid.base0", rel_tol=1e-12)
        for j in range(len(base)):
            _cmp_close(r["base"][j], base[j], "xs.grid.base[%d]" % j, rel_tol=1e-12)
        _cmp_matrix(r["scale"], sc, "xs.grid.scale", rel_tol=1e-9)
        _cmp_matrix(r["sin"], sn, "xs.grid.sin", rel_tol=1e-9)
        _cmp_matrix(r["cos"], cs, "xs.grid.cos", rel_tol=1e-9)
        if ds:
            for i in range(len(sc)):
                for j in range(len(sc[i])):
                    _cmp_close(r["applied_scale"][i][j], 1.0 / sc[i][j],
                               "xs.grid.applied[%d][%d]" % (i, j), rel_tol=1e-12)

    # --- 非法输入：奇数 head_dim 必须报错（官方 rotate_every_two 直接崩）
    for bad in (1, 3, 5, 7):
        try:
            solve("xpos_scale", head_dim=bad, seq_len=3, offset=0, downscale=False)
        except ValueError:
            pass
        else:
            raise AssertionError("xs: odd head_dim %d should be rejected" % bad)
    try:
        solve("xpos_scale", head_dim=4, seq_len=0, offset=0, downscale=False)
    except ValueError:
        pass
    else:
        raise AssertionError("xs: seq_len 0 should be rejected")
    try:
        solve("xpos_scale", head_dim=4, seq_len=3, offset=-1, downscale=False)
    except ValueError:
        pass
    else:
        raise AssertionError("xs: negative offset should be rejected")


# ------------------------------------------------------------------ check 4
def check_multiscale_retention(solve):
    # --- 手算锚点 1：单 head、hidden=2、L=1 的最小可算例子
    X = [[1.0, 2.0]]
    W_Q = [[[1.0, 0.0], [0.0, 1.0]]]
    W_K = [[[1.0, 0.0], [0.0, 1.0]]]
    W_V = [[[1.0, 0.0], [0.0, 1.0]]]
    W_G = [[1.0, 0.0], [0.0, 1.0]]
    W_O = [[1.0, 0.0], [0.0, 1.0]]
    r = solve("multiscale_retention", X=X, W_Q=W_Q, W_K=W_K, W_V=W_V, W_G=W_G, W_O=W_O, heads=1)
    _cmp_close(r["gammas"][0], 0.96875, "ms.a1.gamma0", rel_tol=1e-12)
    # 单 head 时 cat 后就是该 head 的输出
    _cmp_close(r["y_cat"][0][0], r["y_cat"][0][0], "ms.a1.self", rel_tol=0.0)

    # --- 手算锚点 2：GroupNorm 在单组（num_groups=1）时把整行归一化到零均值
    r2 = solve("multiscale_retention", X=X, W_Q=W_Q, W_K=W_K, W_V=W_V, W_G=W_G, W_O=W_O, heads=1)
    y = r2["y_gn"][0]
    mean = sum(y) / len(y)
    _cmp_close(mean, 0.0, "ms.a2.mean_zero", abs_tol=1e-9)
    # y_gn 必须是 y_cat 的 GroupNorm 结果：逐位重算一遍
    expect = oracle_group_norm(r2["y_cat"], 1)
    for d in range(len(y)):
        _cmp_close(y[d], expect[0][d], "ms.a2.gn[%d]" % d, rel_tol=1e-9, abs_tol=1e-9)
    # 归一化后的方差 = raw_var / (raw_var + eps)
    raw = r2["y_cat"][0]
    raw_mean = sum(raw) / len(raw)
    raw_var = sum((v - raw_mean) ** 2 for v in raw) / len(raw)
    got_var = sum((v - mean) ** 2 for v in y) / len(y)
    expect_var = raw_var / (raw_var + _GN_EPS) if raw_var > 0 else 1.0
    _cmp_close(got_var, expect_var, "ms.a2.var", abs_tol=1e-6)

    # --- 手算锚点 3：heads=2 时 gammas 首尾精确
    # hidden 必须能被 heads 整除，且 head_size 必须为偶数（官方 XPOS 的限制）
    X3 = [[1.0, 2.0, 0.5, -1.0]]
    I2 = [[1.0, 0.0], [0.0, 1.0]]
    # 每个 head 的投影是 (hidden=4, head_size=2)
    P2 = [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]
    Q2 = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
    W_Q2 = [P2, Q2]
    W_K2 = [P2, Q2]
    W_V2 = [P2, Q2]
    I4 = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
    r3 = solve("multiscale_retention", X=X3, W_Q=W_Q2, W_K=W_K2, W_V=W_V2, W_G=I4, W_O=I4, heads=2)
    _cmp_close(r3["gammas"][0], 1.0 - 1.0 / 32.0, "ms.a3.gamma_lo", rel_tol=1e-12)
    _cmp_close(r3["gammas"][1], 1.0 - 1.0 / 512.0, "ms.a3.gamma_hi", rel_tol=1e-12)

    # --- 网格对拍
    rng = _Rng(8444)
    for _ in range(30):
        heads = rng.randint(1, 4)
        head_size = 2 * rng.randint(1, 2)   # head_size 必须为偶数（官方 rotate_every_two 的限制）
        hidden = heads * head_size
        L = rng.randint(1, 6)
        def rm(rows, cols):
            return [[rng.randint(-4, 4) / 2.0 for _ in range(cols)] for _ in range(rows)]
        X = rm(L, hidden)
        W_Q = [rm(hidden, head_size) for _ in range(heads)]
        W_K = [rm(hidden, head_size) for _ in range(heads)]
        W_V = [rm(hidden, head_size) for _ in range(heads)]
        W_G = rm(hidden, hidden)
        W_O = rm(hidden, hidden)
        r = solve("multiscale_retention", X=X, W_Q=W_Q, W_K=W_K, W_V=W_V, W_G=W_G, W_O=W_O, heads=heads)
        gammas, y_cat, y_gn, out = oracle_multiscale(X, W_Q, W_K, W_V, W_G, W_O, heads)
        for i in range(heads):
            _cmp_close(r["gammas"][i], gammas[i], "ms.grid.gamma[%d]" % i, rel_tol=1e-12)
        _cmp_matrix(r["y_cat"], y_cat, "ms.grid.y_cat", rel_tol=1e-9, abs_tol=1e-8)
        _cmp_matrix(r["y_gn"], y_gn, "ms.grid.y_gn", rel_tol=1e-9, abs_tol=1e-6)
        _cmp_matrix(r["out"], out, "ms.grid.out", rel_tol=1e-9, abs_tol=1e-6)

    # --- 非法输入
    try:
        solve("multiscale_retention", X=[], W_Q=[], W_K=[], W_V=[], W_G=[], W_O=[], heads=1)
    except ValueError:
        pass
    else:
        raise AssertionError("ms: empty X should be rejected")
    try:
        solve("multiscale_retention", X=[[1.0, 2.0, 3.0]], W_Q=[[[1.0]]], W_K=[[[1.0]]], W_V=[[[1.0]]], W_G=[[1.0]], W_O=[[1.0]], heads=2)
    except ValueError:
        pass
    else:
        raise AssertionError("ms: hidden not divisible by heads should be rejected")
    # hidden=5、heads=2 -> hidden % heads = 1（不可整除），但 head_size = 5//2 = 2（正偶数）。
    # 这一条专门盯住「不可整除」这道防线：若它被削弱，head_size 的偶数校验不会兜住。
    try:
        solve("multiscale_retention", X=[[1.0, 2.0, 3.0, 4.0, 5.0]],
               W_Q=[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.0], [0.0, 0.5]],
                     [[0.0, 1.0], [1.0, 0.0], [0.0, 0.5], [0.5, 0.5], [1.0, 0.0]]],
               W_K=[[[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.5, 0.0], [1.0, 0.5]],
                     [[0.0, 1.0], [1.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.5, 1.0]]],
               W_V=[[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
                     [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [1.0, 0.0], [0.0, 0.5]]],
               W_G=[[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0]],
               W_O=[[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0]], heads=2)
    except ValueError:
        pass
    else:
        raise AssertionError("ms: hidden not divisible by heads (even head_size) should be rejected")
    try:
        solve("multiscale_retention", X=[[1.0, 2.0]], W_Q=[[[1.0, 0.0], [0.0, 1.0]]], W_K=[[[1.0, 0.0], [0.0, 1.0]]], W_V=[[[1.0, 0.0], [0.0, 1.0]]], W_G=[[1.0, 0.0], [0.0, 1.0]], W_O=[[1.0, 0.0], [0.0, 1.0]], heads=0)
    except ValueError:
        pass
    else:
        raise AssertionError("ms: heads 0 should be rejected")
    # hidden=2、heads=1 -> head_size=2（偶数，合法）
    # hidden=3、heads=1 -> head_size=3（奇数，官方 XPOS 会崩，必须拒绝）
    try:
        solve("multiscale_retention", X=[[1.0, 2.0, 3.0]],
               W_Q=[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]],
               W_K=[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]],
               W_V=[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]],
               W_G=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
               W_O=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], heads=1)
    except ValueError:
        pass
    else:
        raise AssertionError("ms: odd head_size should be rejected")
    # W_Q 的 head 数与 heads 不一致
    try:
        solve("multiscale_retention", X=[[1.0, 2.0]],
               W_Q=[[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]],
               W_K=[[[1.0, 0.0], [0.0, 1.0]]], W_V=[[[1.0, 0.0], [0.0, 1.0]]],
               W_G=[[1.0, 0.0], [0.0, 1.0]], W_O=[[1.0, 0.0], [0.0, 1.0]], heads=1)
    except ValueError:
        pass
    else:
        raise AssertionError("ms: mismatched W_Q head count should be rejected")
    # W_Q 的行数与 hidden 不一致
    try:
        solve("multiscale_retention", X=[[1.0, 2.0]],
               W_Q=[[[1.0, 0.0]]], W_K=[[[1.0, 0.0], [0.0, 1.0]]], W_V=[[[1.0, 0.0], [0.0, 1.0]]],
               W_G=[[1.0, 0.0], [0.0, 1.0]], W_O=[[1.0, 0.0], [0.0, 1.0]], heads=1)
    except ValueError:
        pass
    else:
        raise AssertionError("ms: mismatched W_Q rows should be rejected")


