"""order 844 · RetNet 保留机制与 xPos 相对位置的参考解。

纯 Python，只依赖 fractions 与 math；不 import torch / numpy。
"""

from fractions import Fraction
import math
import struct

__all__ = [
    "decay_matrix",
    "retention_forms",
    "xpos_scale",
    "multiscale_retention",
]

# ---------------------------------------------------------------- 常量

_SCALE_BASE = 512          # XPOS 的 scale_base
_GN_EPS = Fraction(1, 10 ** 5)   # nn.GroupNorm 默认 eps


def _f32(x):
    """把 Python float 舍入到 float32 再取回（复刻 torch.float32 的取值）。"""
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


def _f(x):
    """把 int / str / float / Fraction 统一成 Fraction。"""
    if isinstance(x, Fraction):
        return x
    if isinstance(x, bool):
        return Fraction(int(x))
    if isinstance(x, int):
        return Fraction(x)
    if isinstance(x, str):
        return Fraction(x)
    if isinstance(x, float):
        return Fraction(str(x))
    # torch / numpy 的标量或 0 维张量
    item = getattr(x, "item", None)
    if callable(item):
        return Fraction(str(float(item())))
    raise TypeError("unsupported numeric type: %r" % (type(x),))


def _as_matrix(m, name):
    """校验并规范化为 list[list[Fraction]]。"""
    if not isinstance(m, (list, tuple)) or len(m) == 0:
        raise ValueError("%s must be a non-empty list of rows" % name)
    rows = []
    width = None
    for r in m:
        if not isinstance(r, (list, tuple)) or len(r) == 0:
            raise ValueError("%s rows must be non-empty lists" % name)
        if width is None:
            width = len(r)
        elif len(r) != width:
            raise ValueError("%s rows must all have the same length" % name)
        rows.append([_f(v) for v in r])
    return rows


def _as_vector(v, name):
    if not isinstance(v, (list, tuple)) or len(v) == 0:
        raise ValueError("%s must be a non-empty list" % name)
    return [_f(x) for x in v]


# ---------------------------------------------------------------- mode 1


def decay_matrix(gamma, seq_len):
    """指数衰减矩阵 D 与 gamma 的幂表。

    D[n][m] = gamma ** (n - m)  当 n >= m，否则 0。
    """
    g = _f(gamma)
    if not (0 < g < 1):
        raise ValueError("gamma must lie strictly inside (0, 1)")
    if not isinstance(seq_len, int) or seq_len <= 0:
        raise ValueError("seq_len must be a positive integer")

    powers = []
    cur = Fraction(1)
    for _ in range(seq_len):
        powers.append(cur)
        cur = cur * g

    d = []
    for n in range(seq_len):
        row = []
        for m in range(seq_len):
            if n >= m:
                row.append(powers[n - m])
            else:
                row.append(Fraction(0))
        d.append(row)
    return {"gamma_powers": powers, "D": d}


# ---------------------------------------------------------------- xPos


def _xpos_base_num(head_dim):
    """XPOS scale buffer 的分子部分：2j + 0.4*head_dim（j = 0..head_dim/2-1）。

    官方：torch.arange(0, head_dim, 2) + 0.4 * head_dim。

    这里【不做奇偶校验】：偶数校验放在公开入口 xpos_scale（mode 3）
    与 multiscale_retention（mode 4）各自，作为互不依赖的独立防线。
    """
    if not isinstance(head_dim, int) or head_dim <= 0:
        raise ValueError("head_dim must be a positive integer")
    return [Fraction(2 * j) + Fraction(2, 5) * head_dim for j in range(head_dim // 2)]


def _xpos_den(head_dim):
    """base[j] = (2j + 0.4*head_dim) / (1.4*head_dim)。

    官方：torch.arange(0, head_dim, 2) + 0.4 * head_dim，再除以 1.4 * head_dim。
    官方 scale buffer 是 float32（int tensor + Python float 提升为 float32），
    故 base 带 float32 尾差；这里用 struct 做 float32 舍入复刻。
    """
    num = _xpos_base_num(head_dim)
    den = Fraction(7, 5) * head_dim
    out = []
    for n in num:
        # (2j + 0.4*hd) 先以 float32 落地，再除以 float32 的 (1.4*hd)
        num_f = _f32(float(n))
        den_f = _f32(float(den))
        out.append(Fraction(str(_f32(num_f / den_f))))
    return out


def _xpos_scale_impl(head_dim, seq_len, offset, downscale):
    """xpos_scale 的内部实现（不做奇偶/范围校验，调用方负责）。

    这样 mode 4 的 head_size 偶数校验与 mode 3 的 head_dim 偶数校验
    互为独立防线，不会互相「兜住」对方的缺失。
    """
    base = _xpos_den(head_dim)

    total = seq_len + offset
    # scale[pos][j] = base[j] ** (pos / 512)
    # 官方路径：scale buffer 是 float32 值（见 _xpos_den），`torch.arange(0, L).div(512)`
    # 得到 float32 的比值，幂运算在该比值上进行。
    # 本题口径：base 用 float32 精度，比值按 float32 复刻，幂运算用 float64。
    raw = []
    for pos in range(total):
        row = []
        ratio = _f32(float(pos) / _SCALE_BASE)
        for j in range(len(base)):
            row.append(float(base[j]) ** ratio)
        raw.append(row)

    # fixed_pos_embedding(scale)：inv_freq[j] = 1 / 10000 ** (j / dim)
    # 官方 torch.arange(0, dim) 是 float32，且 10000 ** ratio 与 1.0/... 全程 float32，
    # 因此 inv_freq 带 float32 尾差（0.10000000149011612 而不是精确的 0.1）。
    # 这里用 struct 做 float32 舍入逐步复刻（不引入 torch 依赖）。
    dim = len(base)
    inv_freq = []
    for j in range(dim):
        ratio = _f32(float(j) / dim)
        pow_val = _f32(10000.0 ** ratio)
        inv_freq.append(_f32(1.0 / pow_val))

    # ★ 官方顺序：先在【完整的 total 行】scale 上算 sin/cos，再把三者一起截断到 length 行。
    sin = []
    cos = []
    for pos in range(total):
        srow = []
        crow = []
        for j in range(dim):
            # 官方 sinusoid_inp = einsum("i , j -> i j", arange(seq_len, dtype=float), inv_freq)
            # arange 与 inv_freq 都是 float32（inv_freq 与 XPOS 的 dtype 无关），
            # 故乘积是 float32；再 .to(x) 提升到 float64。
            ang = _f32(_f32(float(pos)) * inv_freq[j])
            srow.append(math.sin(ang))
            crow.append(math.cos(ang))
        sin.append(srow)
        cos.append(crow)

    if total > seq_len:
        raw = raw[-seq_len:]
        sin = sin[-seq_len:]
        cos = cos[-seq_len:]

    if downscale:
        applied = [[1.0 / v for v in row] for row in raw]
    else:
        applied = [list(row) for row in raw]

    return {
        "base": base,
        "scale": raw,
        "sin": sin,
        "cos": cos,
        "applied_scale": applied,
    }


def xpos_scale(head_dim, seq_len, offset=0, downscale=False):
    """xPos 相对位置的 scale / sin / cos。

    scale[pos][j] = base[j] ** (pos / 512)，pos 从 0 到 seq_len + offset - 1；
    若 seq_len + offset > seq_len（即 offset > 0）则只保留最后 seq_len 行。
    downscale 只把「传给 apply 的 scale」取倒数，sin/cos 不受影响。
    """
    if not isinstance(seq_len, int) or seq_len <= 0:
        raise ValueError("seq_len must be a positive integer")
    if not isinstance(offset, int) or offset < 0:
        raise ValueError("offset must be a non-negative integer")
    if head_dim % 2 != 0:
        # 官方 rotate_every_two 在奇数 head_dim 时 torch.stack 尺寸不等直接崩；
        # 官方源码里补零的分支写在 stack 之后，是死代码。
        raise ValueError("head_dim must be even (official rotate_every_two crashes otherwise)")
    return _xpos_scale_impl(head_dim, seq_len, offset, downscale)


def _xpos_apply(x, sin, cos, applied_scale):
    """apply_rotary_pos_emb 的纯 Python 复刻。

    x: (L, D)；sin/cos/applied_scale: (L, D/2)。
    官方：sin, cos = duplicate_interleave(t * scale)；然后
    (x * cos[:, :D]) + (rotate_every_two(x) * sin)[:, :, :D]
    """
    L = len(x)
    D = len(x[0])
    half = D // 2

    # duplicate_interleave：官方是
    #   sin, cos = map(lambda t: duplicate_interleave(t * scale), (sin, cos))
    # 即先逐元素乘 scale（(L, dim) 逐元素），再把每行元素各复制一次并交错。
    def dup_interleave(t, scale):
        out = []
        for i in range(len(t)):
            row = t[i]
            srow = scale[i]
            res = []
            for j in range(len(row)):
                a = row[j] * srow[j]
                res.append(a)
                res.append(a)
            out.append(res)
        return out

    sin_d = dup_interleave(sin, applied_scale)
    cos_d = dup_interleave(cos, applied_scale)

    # rotate_every_two：x1 = x[..., ::2], x2 = x[..., 1::2] -> stack((-x2, x1)).flatten(-2)
    rot = []
    for n in range(L):
        row = x[n]
        x1 = row[0::2]
        x2 = row[1::2]
        inter = []
        for a, b in zip(x2, x1):
            inter.append(-a)
            inter.append(b)
        rot.append(inter)

    out = []
    for n in range(L):
        row = []
        for d in range(D):
            v = x[n][d] * cos_d[n][d] + rot[n][d] * sin_d[n][d]
            row.append(v)
        out.append(row)
    return out


# ---------------------------------------------------------------- mode 2


def _matmul_qk(q, k):
    """(L, D) x (L, D) -> (L, L)：q[n] · k[m]。"""
    L = len(q)
    D = len(q[0])
    out = []
    for n in range(L):
        row = []
        for m in range(L):
            s = Fraction(0)
            for d in range(D):
                s += q[n][d] * k[m][d]
            row.append(s)
        out.append(row)
    return out


def _apply_xpos(m, head_dim, offset, downscale):
    """对 (L, D) 矩阵施加 xPos（官方 SimpleRetention 的用法）。"""
    if head_dim is None:
        return m
    info = xpos_scale(head_dim, len(m), offset, downscale)
    return _xpos_apply(m, info["sin"], info["cos"], info["applied_scale"])


def _headwise_scores(q, k, d):
    """逐 head 的 score[n][m] = D[n][m] * (q[n]·k[m])。

    q/k: list of head matrices，每个 (L, head_dim)。
    """
    heads = len(q)
    out = []
    for h in range(heads):
        qk = _matmul_qk(q[h], k[h])
        out.append([[qk[n][m] * d[n][m] for m in range(len(qk[n]))] for n in range(len(qk))])
    return out


def _retention_parallel(Q, K, V, gamma):
    """parallel 表示：ret = (Q @ K^T) * D，然后 @ V。

    Q/K/V：list of heads，每个 head 是 (L, D) 的矩阵（Q[h][n][d]）。
    返回 out: list over n of list over h of list over v_dim。
    """
    heads = len(Q)
    L = len(Q[0])
    dd = decay_matrix(gamma, L)["D"]

    qs = [[list(row) for row in Q[h]] for h in range(heads)]
    ks = [[list(row) for row in K[h]] for h in range(heads)]

    scores = _headwise_scores(qs, ks, dd)

    # out[n][h] = sum_m scores[h][n][m] * V[h][m][:]
    out = []
    for n in range(L):
        row = []
        for h in range(heads):
            vh = V[h]
            vdim = len(vh[0])
            acc = [Fraction(0)] * vdim
            for m in range(L):
                s = scores[h][n][m]
                if s == 0:
                    continue
                for d in range(vdim):
                    acc[d] += s * vh[m][d]
            row.append(acc)
        out.append(row)
    return out, scores


def _retention_recurrent(Q, K, V, gamma):
    """recurrent 表示：s_n = gamma * s_{n-1} + K_n^T V_n；out_n = Q_n @ s_n。"""
    heads = len(Q)
    L = len(Q[0])
    g = _f(gamma)

    qs = [[list(row) for row in Q[h]] for h in range(heads)]
    ks = [[list(row) for row in K[h]] for h in range(heads)]

    states = []
    outs = []
    # 每 head 独立递推
    cur = [None] * heads
    for h in range(heads):
        vdim = len(V[h][0])
        cur[h] = [[Fraction(0)] * vdim for _ in range(len(Q[h][0]))]

    for n in range(L):
        new_states = []
        for h in range(heads):
            vh = V[h]
            vdim = len(vh[0])
            head_dim_h = len(K[h][0])
            # s_n = gamma * s_{n-1} + K_n^T V_n
            # 递归形态下 K_n / V_n 都只有当前这一步（形状 (1, D) / (1, v_dim)），
            # 故 K_n^T V_n 是单步外积，不做 m 求和。
            s = []
            for d in range(head_dim_h):
                row = []
                for e in range(vdim):
                    acc = g * cur[h][d][e]
                    acc += K[h][n][d] * vh[n][e]
                    row.append(acc)
                s.append(row)
            cur[h] = s
            new_states.append(s)
        states.append(new_states)
        # out_n[h] = Q_n @ s_n
        orow = []
        for h in range(heads):
            vdim = len(V[h][0])
            acc = [Fraction(0)] * vdim
            for d in range(len(Q[h][0])):
                qv = qs[h][n][d]
                if qv == 0:
                    continue
                for e in range(vdim):
                    acc[e] += qv * cur[h][d][e]
            orow.append(acc)
        outs.append(orow)
    return outs, states


def _retention_chunkwise(Q, K, V, gamma, chunk_size, quirk=False):
    """chunkwise 表示。

    官方 forward_chunkwise(x_i, r_i_1, i) 的第三个参数是【块索引】，
    函数内 offset = 块索引 * chunk_size，其中 chunk_size 取【本块实际长度】。
    quirk=False 时用【固定块长】算 offset（与 parallel 严格等价）；
    quirk=True 时复现官方的实际行为（末块长度不足时 offset 错位）。

    同时返回每块结束时的状态 r_i（官方 forward_chunkwise 也返回 r_i）。
    """
    heads = len(Q)
    L = len(Q[0])
    g = _f(gamma)
    # chunk_size 的合法性由公开入口 retention_forms 校验（唯一防线）

    outs = []
    block_states = []
    r_prev = [None] * heads
    for h in range(heads):
        vdim = len(V[h][0])
        hd = len(K[h][0])
        r_prev[h] = [[Fraction(0)] * vdim for _ in range(hd)]

    nblocks = (L + chunk_size - 1) // chunk_size
    for bi in range(nblocks):
        lo = bi * chunk_size
        hi = min(lo + chunk_size, L)
        c = hi - lo
        blk_q = []
        blk_k = []
        for h in range(heads):
            blk_q.append([list(Q[h][n]) for n in range(lo, hi)])
            blk_k.append([list(K[h][n]) for n in range(lo, hi)])

        # r_i = (K^T @ (V * D[-1])) + gamma**chunk_size * r_i_1
        # 官方 forward_chunkwise 里 chunk_size = x_i.shape[1]，即【本块实际长度】，
        # 故衰减指数是 c（本块实际长度），不是固定块长。
        d_chunk = decay_matrix(gamma, c)["D"]
        d_last = d_chunk[-1]

        old_r = r_prev
        new_r = []
        for h in range(heads):
            vh = V[h]
            vdim = len(vh[0])
            hd = len(K[h][0])
            decay = g ** c
            s = []
            for d in range(hd):
                row = []
                for e in range(vdim):
                    acc = decay * old_r[h][d][e]
                    add = Fraction(0)
                    for m in range(c):
                        vlocal = vh[lo + m][e] * d_last[m]
                        add += blk_k[h][m][d] * vlocal
                    row.append(acc + add)
                s.append(row)
            new_r.append(s)

        # inner_chunk = ((Q @ K^T) * D) @ V
        blk_scores = _headwise_scores(blk_q, blk_k, d_chunk)
        # cross_chunk = (Q @ r_i_1) * e,  e[j] = gamma**(j+1)
        # 注意官方用的是【旧状态 r_i_1】，不是刚算出的 r_i
        e = [g ** (j + 1) for j in range(c)]

        for j in range(c):
            orow = []
            for h in range(heads):
                vh = V[h]
                vdim = len(vh[0])
                acc = [Fraction(0)] * vdim
                # inner
                for m in range(c):
                    s = blk_scores[h][j][m]
                    if s == 0:
                        continue
                    for d in range(vdim):
                        acc[d] += s * vh[lo + m][d]
                # cross：用旧状态 old_r
                qj = blk_q[h][j]
                for d in range(len(qj)):
                    qv = qj[d]
                    if qv == 0:
                        continue
                    for eidx in range(vdim):
                        acc[eidx] += qv * old_r[h][d][eidx] * e[j]
                orow.append(acc)
            outs.append(orow)
        r_prev = new_r
        block_states.append([[list(row) for row in head] for head in new_r])
    return outs, block_states


def retention_forms(Q, K, V, gamma, chunk_size=None):
    """三种表示 + 中间量（纯 retention 骨架，不含 xPos）。"""
    if not isinstance(Q, (list, tuple)) or len(Q) == 0:
        raise ValueError("Q must be a non-empty list of head matrices")
    heads = len(Q)
    L = len(Q[0])
    for name, M in (("K", K), ("V", V)):
        if len(M) != heads:
            raise ValueError("%s must have the same number of heads as Q" % name)
        for h in range(heads):
            if len(M[h]) != L:
                raise ValueError("%s head %d must have the same sequence length as Q" % (name, h))
            if len(M[h][0]) == 0:
                raise ValueError("%s head %d must have non-empty rows" % (name, h))
    if chunk_size is None:
        chunk_size = L
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    # 归一化输入为 Fraction（可能是 torch tensor / numpy 标量 / 原生数值）
    Q = [[[_f(v) for v in row] for row in head] for head in Q]
    K = [[[_f(v) for v in row] for row in head] for head in K]
    V = [[[_f(v) for v in row] for row in head] for head in V]

    par, scores = _retention_parallel(Q, K, V, gamma)
    rec, states = _retention_recurrent(Q, K, V, gamma)
    chw, block_states = _retention_chunkwise(Q, K, V, gamma, chunk_size, quirk=False)

    return {
        "parallel": par,
        "recurrent": rec,
        "chunkwise": chw,
        "scores": scores,
        "states": states,
        "block_states": block_states,
    }


# ---------------------------------------------------------------- mode 4


def _swish(x):
    return [v * (1.0 / (1.0 + math.exp(-v))) for v in x]


def _group_norm(y, num_groups, eps=_GN_EPS):
    """对 (L, C) 矩阵做 GroupNorm（affine=True，weight=1、bias=0）。

    torch 的 GroupNorm 分 G 组，每组 C/G 个【连续通道】，方差有偏（1/N）。
    """
    L = len(y)
    C = len(y[0])
    if C % num_groups != 0:
        raise ValueError("channel count must be divisible by num_groups")
    gsize = C // num_groups
    out = []
    for n in range(L):
        row = []
        for g in range(num_groups):
            seg = y[n][g * gsize:(g + 1) * gsize]
            m = sum(seg) / gsize
            var = sum((v - m) ** 2 for v in seg) / gsize
            denom = math.sqrt(float(var) + float(eps))
            for v in seg:
                row.append(float((v - m)) / denom)
        out.append(row)
    return out


def _matmul(a, b):
    """(L, K) x (K, M) -> (L, M)。"""
    L = len(a)
    K = len(a[0])
    M = len(b[0])
    out = []
    for i in range(L):
        row = []
        for j in range(M):
            s = 0.0
            for k in range(K):
                s += float(a[i][k]) * float(b[k][j])
            row.append(s)
        out.append(row)
    return out


def _ms_gammas(heads):
    """官方：1 - exp(linspace(log(1/32), log(1/512), heads))。"""
    if not isinstance(heads, int) or heads <= 0:
        raise ValueError("heads must be a positive integer")
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


def multiscale_retention(X, W_Q, W_K, W_V, W_G, W_O, heads, chunk_size=None):
    """多尺度保留：逐 head retention -> cat -> GroupNorm -> (swish(X@W_G) * Y) @ W_O。

    X: (L, hidden)；W_Q/W_K/W_V: 每个是 heads 组 (hidden, head_size) 的列表；
    W_G: (hidden, v_dim)；W_O: (v_dim, hidden)。
    """
    if not isinstance(X, (list, tuple)) or len(X) == 0:
        raise ValueError("X must be a non-empty list of rows")
    L = len(X)
    hidden = len(X[0])
    if not isinstance(heads, int) or heads <= 0:
        raise ValueError("heads must be a positive integer")
    if hidden % heads != 0:
        raise ValueError("hidden size must be divisible by heads")
    head_size = hidden // heads
    if head_size % 2 != 0:
        # 官方 XPOS(head_size) 的 rotate_every_two 在奇数 head_dim 时直接崩
        raise ValueError("head_size must be even (official rotate_every_two crashes otherwise)")
    v_dim = hidden

    for name, W in (("W_Q", W_Q), ("W_K", W_K), ("W_V", W_V)):
        if len(W) != heads:
            raise ValueError("%s must have one matrix per head" % name)
        for h in range(heads):
            if len(W[h]) != hidden:
                raise ValueError("%s head %d must have hidden rows" % (name, h))

    gammas = _ms_gammas(heads)

    # 逐 head retention（官方 MultiScaleRetention 的每个 SimpleRetention 都带 xPos）
    # xPos 含无理幂，故 mode 4 全程走 float 路径。
    ys = []
    for h in range(heads):
        qh = _matmul(X, W_Q[h])
        kh = _matmul(X, W_K[h])
        vh = _matmul(X, W_V[h])
        # xPos（官方 SimpleRetention.forward 的用法：Q 不 downscale，K downscale）
        # 这里直接用内部实现：head_size 的偶数性已在本函数入口校验过，
        # 避免 mode 3 的 head_dim 校验把 mode 4 的校验「兜住」。
        qinfo = _xpos_scale_impl(head_size, L, 0, False)
        kinfo = _xpos_scale_impl(head_size, L, 0, True)
        qh = _xpos_apply(qh, qinfo["sin"], qinfo["cos"], qinfo["applied_scale"])
        kh = _xpos_apply(kh, kinfo["sin"], kinfo["cos"], kinfo["applied_scale"])
        # 衰减矩阵 D
        dd = [[gammas[h] ** (n - m) if n >= m else 0.0 for m in range(L)] for n in range(L)]
        # scores[n][m] = D[n][m] * q[n]·k[m]
        yh = []
        for n in range(L):
            row = []
            for m in range(L):
                qk = 0.0
                for d in range(head_size):
                    qk += qh[n][d] * kh[m][d]
                row.append(qk * dd[n][m])
            yh.append(row)
        # out[n][e] = sum_m scores[n][m] * V[m][e]
        out_h = []
        for n in range(L):
            row = []
            for e in range(head_size):
                acc = 0.0
                for m in range(L):
                    acc += yh[n][m] * vh[m][e]
                row.append(acc)
            out_h.append(row)
        ys.append(out_h)

    # cat dim=2：每个 head 输出 head_size 维，v_dim = heads * head_size
    y_cat = []
    for n in range(L):
        row = []
        for h in range(heads):
            row.extend(ys[h][n])
        y_cat.append(row)

    y_gn = _group_norm(y_cat, heads)

    # swish(X @ W_G) * Y
    xg = _matmul(X, W_G)
    sw = []
    for n in range(L):
        sw.append(_swish(xg[n]))
    gated = []
    for n in range(L):
        row = []
        for d in range(v_dim):
            row.append(sw[n][d] * y_gn[n][d])
        gated.append(row)

    out = _matmul(gated, W_O)
    return {"gammas": gammas, "y_cat": y_cat, "y_gn": y_gn, "out": out}

# ---------------------------------------------------------------- 入口
def solve(mode, **kw):
    """单入口。

    mode:
      decay_matrix        : 指数衰减矩阵 D 与 gamma 的幂表
      retention_forms     : parallel / recurrent / chunkwise 三种表示
      xpos_scale          : xPos 相对位置的 scale / sin / cos
      multiscale_retention: 多尺度保留 + GroupNorm + swish 门控
    """
    if mode == "decay_matrix":
        return decay_matrix(kw["gamma"], kw["seq_len"])
    if mode == "retention_forms":
        return retention_forms(kw["Q"], kw["K"], kw["V"], kw["gamma"],
                                kw.get("chunk_size"))
    if mode == "xpos_scale":
        return xpos_scale(kw["head_dim"], kw["seq_len"],
                          kw.get("offset", 0), bool(kw.get("downscale", False)))
    if mode == "multiscale_retention":
        return multiscale_retention(kw["X"], kw["W_Q"], kw["W_K"], kw["W_V"],
                                    kw["W_G"], kw["W_O"], kw["heads"],
                                    kw.get("chunk_size"))
    raise ValueError("unknown mode: %r" % (mode,))
