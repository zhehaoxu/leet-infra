"""Mamba2 SSD chunk 扫描三段账（纯 Python）。

对应官方源码（state-spaces/mamba, main 分支）：
  mamba_ssm/ops/triton/ssd_chunk_state.py::chunk_state_ref     —— chunk 内状态账
  mamba_ssm/ops/triton/ssd_state_passing.py::state_passing_ref —— chunk 间状态传递
  mamba_ssm/ops/triton/ssd_chunk_scan.py::chunk_scan_ref       —— chunk 内输出账
  mamba_ssm/ops/triton/ssd_combined.py::ssd_chunk_scan_combined_ref —— 三段编排

四个 mode（全部显式接收长度，不依赖任何隐式状态）：
  chunk_state     —— chunk 内每个位置对 chunk 末状态的贡献
  state_passing   —— chunk 之间状态的加权累积传递
  chunk_scan      —— chunk 内输出（交叉项 + 初始状态项 + D 跳连 + z 门控）
  ssd_forward     —— 三段完整编排

数值口径：全程 Python float（float64）。
"""

import math

INF = float("inf")


# ----------------------------------------------------------------- 基础工具
def _dims(*shapes):
    """校验所有嵌套列表形状一致，返回 (b, ...) 各维长度。"""
    if not shapes:
        raise ValueError("empty shape")
    first = shapes[0]
    for other in shapes[1:]:
        if len(other) != len(first):
            raise ValueError("shape rank mismatch")
        for a, b in zip(other, first):
            if a != b:
                raise ValueError("shape element mismatch")
    return tuple(first)


def _nest(x, shape):
    """把嵌套列表规整成给定形状（形状必须匹配）。"""
    def rec(node, sh):
        if not sh:
            if isinstance(node, (list, tuple)):
                raise ValueError("too many elements")
            return float(node)
        if not isinstance(node, (list, tuple)):
            raise ValueError("too few elements")
        if len(node) != sh[0]:
            raise ValueError("length mismatch")
        return [rec(node[i], sh[1:]) for i in range(sh[0])]
    return rec(x, list(shape))


def _flat(x):
    """把任意嵌套列表展平成一维 float 列表。"""
    out = []
    stack = [x]
    while stack:
        node = stack.pop(0)
        if isinstance(node, (list, tuple)):
            stack = list(node) + stack
        else:
            out.append(float(node))
    return out


def _silu(z):
    return z / (1.0 + math.exp(-z)) if z >= -700.0 else 0.0


def _check_shape(node, shape, name):
    """校验嵌套列表形状；任何不符都抛 ValueError（不泄漏 TypeError/IndexError）。"""
    dims = list(shape)
    cur = node
    for i, want in enumerate(dims):
        if not isinstance(cur, (list, tuple)):
            raise ValueError("%s must be a nested list of shape %s" % (name, tuple(dims)))
        if len(cur) != want:
            raise ValueError("%s must be a nested list of shape %s (got dim %d = %d)"
                             % (name, tuple(dims), i, len(cur)))
        cur = cur[0] if want else None
    if dims:
        leaf = node
        for _ in dims:
            leaf = leaf[0]
        if isinstance(leaf, (list, tuple)):
            raise ValueError("%s must be a nested list of shape %s" % (name, tuple(dims)))
    return node


def _check_flat(node, n, name):
    """校验一维列表长度。"""
    if not isinstance(node, (list, tuple)):
        raise ValueError("%s must be a list" % (name,))
    if len(node) != n:
        raise ValueError("%s must have %d elements (got %d)" % (name, n, len(node)))
    for v in node:
        if isinstance(v, (list, tuple)):
            raise ValueError("%s must be a flat list" % (name,))
    return node


def _cumsum(vals):
    out = []
    acc = 0.0
    for v in vals:
        acc += v
        out.append(acc)
    return out


def _reshape(vals, shape):
    """把一维列表按 C 序 reshape 成嵌套列表。"""
    shape = list(shape)
    if not shape:
        return vals[0]
    stride = 1
    for s in shape[1:]:
        stride *= s
    total = 1
    for s in shape:
        total *= s
    if len(vals) != total:
        raise ValueError("reshape size mismatch")
    out = []
    idx = 0

    def build(depth):
        nonlocal idx
        if depth == len(shape):
            v = vals[idx]
            idx += 1
            return v
        return [build(depth + 1) for _ in range(shape[depth])]

    out = build(0)
    return out


def _transpose_last2(m):
    """交换最后两维（用于 b h c l s -> b c h l s 这类 permute）。"""
    return [[list(row) for row in block] for block in m]


# --------------------------------------------------- mode 1: chunk_state_ref
def chunk_state(x, dt, A, B_mat, chunk_size):
    """chunk_state_ref 的等价实现。

    x     : [b][l][h][p]
    dt    : [b][h][c][lc]      （官方 chunk 布局，lc = chunk_size）
    A     : [h]
    B_mat : [b][l][g][n]
    chunk_size : int

    返回 dA_cumsum / decay_states / expanded_B / states 四组账。
    """
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    _check_shape(x, (len(x), len(x[0]), len(x[0][0]), len(x[0][0][0])), "x")
    b = len(x)
    l = len(x[0])
    h = len(x[0][0])
    p = len(x[0][0][0])
    g = len(B_mat[0][0])
    n = len(B_mat[0][0][0])
    if h % g != 0:
        raise ValueError("nheads must be divisible by ngroups")

    c = l // chunk_size
    if c * chunk_size != l:
        raise ValueError("seqlen must be divisible by chunk_size")
    lc = chunk_size

    _check_shape(B_mat, (b, l, g, n), "B_mat")
    _check_shape(dt, (b, h, c, lc), "dt")
    _check_flat(A, h, "A")

    # ---- 官方 repeat(B, "b l g d -> b l (g h) d", h=nheads//ngroups)
    # (g h) 展平顺序：g 在前、h 在后 ⇒ head 索引 = g * (h//G) + hi
    hh = h // g
    expanded_B = []
    for bi in range(b):
        rows = []
        for li in range(l):
            per_head = []
            for gi in range(g):
                src = B_mat[bi][li][gi]
                for _ in range(hh):
                    per_head.append(list(src))
            rows.append(per_head)
        expanded_B.append(rows)

    # ---- dA / dA_cumsum：dt 已是 [b][h][c][lc]
    dt4 = dt

    dA = []
    dA_cumsum = []
    for bi in range(b):
        per_h = []
        per_h_cs = []
        for hi in range(h):
            per_c = []
            per_c_cs = []
            for ci in range(c):
                row = []
                for li in range(lc):
                    row.append(dt4[bi][hi][ci][li] * float(A[hi]))
                per_c.append(row)
                per_c_cs.append(_cumsum(row))
            per_h.append(per_c)
            per_h_cs.append(per_c_cs)
        dA.append(per_h)
        dA_cumsum.append(per_h_cs)

    # ---- decay_states = exp(dA_cumsum[:,:,:,-1:] - dA_cumsum)
    decay_states = []
    for bi in range(b):
        per_h = []
        for hi in range(h):
            per_c = []
            for ci in range(c):
                last = dA_cumsum[bi][hi][ci][lc - 1]
                per_c.append([math.exp(last - v) for v in dA_cumsum[bi][hi][ci]])
            per_h.append(per_c)
        decay_states.append(per_h)

    # ---- states[b,c,h,p,n] = Σ_{l} B5 * decay_states * dt * x5
    #     官方 einsum: "bclhn,bhcl,bhcl,bclhp->bchpn"
    #     求和索引只有 l（h/n/p/b/c 都在输出里）⇒ B5 与 x5 用【同一个】输出 head
    states = []
    for bi in range(b):
        per_c = []
        for ci in range(c):
            per_h = []
            for hi in range(h):
                per_p = []
                for pi in range(p):
                    per_n = []
                    for ni in range(n):
                        acc = 0.0
                        for li in range(lc):
                            ds = decay_states[bi][hi][ci][li]
                            dv = dt4[bi][hi][ci][li]
                            if ds == 0.0:
                                continue
                            bv = expanded_B[bi][ci * lc + li][hi][ni]
                            xv = x[bi][ci * lc + li][hi][pi]
                            acc += bv * ds * dv * xv
                        per_n.append(acc)
                    per_p.append(per_n)
                per_h.append(per_p)
            per_c.append(per_h)
        states.append(per_c)

    return {
        "dA": dA,
        "dA_cumsum": dA_cumsum,
        "decay_states": decay_states,
        "expanded_B": expanded_B,
        "states": states,
        "nchunks": c,
        "chunk_size": lc,
        "dstate": n,
        "headdim": p,
    }


# ---------------------------------------------- mode 2: state_passing_ref
def state_passing(states, dA_chunk_cumsum, initial_states=None):
    """state_passing_ref 的等价实现。

    states            : [b][c][h][d]      （d = headdim*dstate 的展平长度）
    dA_chunk_cumsum   : [b][h][c]
    initial_states    : [b][h][d] 或 None

    返回 decay_chunk / passed / final_state。
    """
    b = len(states)
    c = len(states[0])
    h = len(states[0][0])
    d = len(states[0][0][0])
    _check_shape(states, (b, c, h, d), "states")
    _check_shape(dA_chunk_cumsum, (b, h, c), "dA_chunk_cumsum")

    # 官方：if initial_states is None: initial_states = torch.zeros_like(states[:, 0])
    if initial_states is None:
        init = [[[0.0] * d for _ in range(h)] for _ in range(b)]
    else:
        init = initial_states
        _check_shape(init, (b, h, d), "initial_states")

    # 官方：states = cat([rearrange(initial_states,"b h d -> b 1 h d"), states], dim=1)
    # ⇒ 长度从 c 变成 c+1，第 0 项是初始状态
    cat_states = []
    for bi in range(b):
        per_c = [[list(init[bi][hi]) for hi in range(h)]]
        for ci in range(c):
            per_c.append([list(states[bi][ci][hi]) for hi in range(h)])
        cat_states.append(per_c)

    # 官方：dA_chunk_cumsum = F.pad(dA_chunk_cumsum, (1,0)); cumsum(dim=-1)
    dacc = []
    for bi in range(b):
        per_h = []
        for hi in range(h):
            row = [0.0] + [float(dA_chunk_cumsum[bi][hi][ci]) for ci in range(c)]
            per_h.append(_cumsum(row))
        dacc.append(per_h)
    nc_pad = c + 1

    # 官方：dt_chunk_segment_sum = dacc[:,:,:,None] - dacc[:,:,None,:]
    decay_chunk = []
    for bi in range(b):
        per_h = []
        for hi in range(h):
            per_z = []
            for zi in range(nc_pad):
                row = []
                for zi2 in range(nc_pad):
                    v = dacc[bi][hi][zi] - dacc[bi][hi][zi2]
                    e = math.exp(v) if v < 700.0 else INF
                    # 官方 masked_fill(~tril, 0) ⇒ 上三角清零
                    row.append(e if zi2 <= zi else 0.0)
                per_z.append(row)
            per_h.append(per_z)
        decay_chunk.append(per_h)

    # 官方：out = einsum("bhzc,bchd->bzhd", decay_chunk, states)
    # ⇒ out[b,z,h,d] = Σ_c decay_chunk[b,h,z,c] * cat_states[b,c,h,d]
    #   （行 z 固定、列 c 遍历求和；decay_chunk 第 0 列对应初始状态那一项）
    out = []
    for bi in range(b):
        per_z = []
        for zi in range(nc_pad):
            per_h = []
            for hi in range(h):
                per_d = []
                for di in range(d):
                    acc = 0.0
                    for zc in range(nc_pad):
                        w = decay_chunk[bi][hi][zi][zc]
                        if w == 0.0:
                            continue
                        acc += w * cat_states[bi][zc][hi][di]
                    per_d.append(acc)
                per_h.append(per_d)
            per_z.append(per_h)
        out.append(per_z)

    return {
        "decay_chunk": decay_chunk,
        "passed": [[out[bi][zi] for zi in range(c)] for bi in range(b)],
        "final_state": [out[bi][nc_pad - 1] for bi in range(b)],
        "nchunks": c,
    }


# ------------------------------------------------- mode 3: chunk_scan_ref
def chunk_scan(B_mat, C_mat, x, dt, dA_cumsum, prev_states, D=None, z=None):
    """chunk_scan_ref 的等价实现。

    B_mat / C_mat : [b][l][g][n]
    x              : [b][l][h][p]
    dt             : [b][h][c][lc]
    dA_cumsum      : [b][h][c][lc]
    prev_states    : [b][c][h][p][n]
    D              : [h] 或 [h][p] 或 None
    z              : [b][l][h][p] 或 None
    """
    b = len(x)
    l = len(x[0])
    h = len(x[0][0])
    p = len(x[0][0][0])
    g = len(B_mat[0][0])
    n = len(B_mat[0][0][0])
    if h % g != 0:
        raise ValueError("nheads must be divisible by ngroups")
    c = len(dt[0][0])
    lc = len(dt[0][0][0])
    if c * lc != l:
        raise ValueError("seqlen must equal nchunks * chunk_size")
    _check_shape(x, (b, l, h, p), "x")
    _check_shape(B_mat, (b, l, g, n), "B_mat")
    _check_shape(C_mat, (b, l, g, n), "C_mat")
    _check_shape(dt, (b, h, c, lc), "dt")
    _check_shape(dA_cumsum, (b, h, c, lc), "dA_cumsum")
    _check_shape(prev_states, (b, c, h, p, n), "prev_states")

    hh = h // g

    def expand(mat):
        out = []
        for bi in range(b):
            rows = []
            for li in range(l):
                per_head = []
                for gi in range(g):
                    src = mat[bi][li][gi]
                    for _ in range(hh):
                        per_head.append(list(src))
                rows.append(per_head)
            out.append(rows)
        return out

    B_exp = expand(B_mat)
    C_exp = expand(C_mat)

    # ---- CB[b,c,h,l,s] = Σ_n C5[b,c,l,h,n] * B5[b,c,s,h,n]
    CB = []
    for bi in range(b):
        per_c = []
        for ci in range(c):
            per_h = []
            for hi in range(h):
                per_l = []
                for li in range(lc):
                    row = []
                    for si in range(lc):
                        acc = 0.0
                        for ni in range(n):
                            cv = C_exp[bi][ci * lc + li][hi][ni]
                            bv = B_exp[bi][ci * lc + si][hi][ni]
                            acc += cv * bv
                        row.append(acc)
                    per_l.append(row)
                per_h.append(per_l)
            per_c.append(per_h)
        per_c_ = per_c
        CB.append(per_c_)

    # ---- dt_segment_sum / decay / scores_decay
    decay = []
    for bi in range(b):
        per_h = []
        for hi in range(h):
            per_c = []
            for ci in range(c):
                per_l = []
                for li in range(lc):
                    row = []
                    for si in range(lc):
                        v = dA_cumsum[bi][hi][ci][li] - dA_cumsum[bi][hi][ci][si]
                        row.append(math.exp(v) if v < 700.0 else INF)
                    per_l.append(row)
                per_c.append(per_l)
            per_h.append(per_c)
        decay.append(per_h)

    # 官方：rearrange(decay, "b h c l s -> b c h l s")
    decay_p = []
    for bi in range(b):
        per_c = []
        for ci in range(c):
            per_h = []
            for hi in range(h):
                per_h.append(decay[bi][hi][ci])
            per_c.append(per_h)
        decay_p.append(per_c)

    scores_decay = []
    for bi in range(b):
        per_c = []
        for ci in range(c):
            per_h = []
            for hi in range(h):
                per_l = []
                for li in range(lc):
                    row = []
                    for si in range(lc):
                        v = CB[bi][ci][hi][li][si] * decay_p[bi][ci][hi][li][si]
                        # 官方 masked_fill(~tril(Lc), 0)
                        row.append(v if si <= li else 0.0)
                    per_l.append(row)
                per_h.append(per_l)
            per_c.append(per_h)
        scores_decay.append(per_c)

    # ---- out_intra[b,c,l,h,p] = Σ_s scores_decay * dt[b,h,c,s] * x[b,c,s,h,p]
    out_intra = []
    for bi in range(b):
        per_c = []
        for ci in range(c):
            per_l = []
            for li in range(lc):
                per_h = []
                for hi in range(h):
                    row = []
                    for pi in range(p):
                        acc = 0.0
                        for si in range(lc):
                            sd = scores_decay[bi][ci][hi][li][si]
                            if sd == 0.0:
                                continue
                            dv = dt[bi][hi][ci][si]
                            xv = x[bi][ci * lc + si][hi][pi]
                            acc += sd * dv * xv
                        row.append(acc)
                    per_h.append(row)
                per_l.append(per_h)
            per_c.append(per_l)
        out_intra.append(per_c)

    # ---- state_decay_out = exp(dA_cumsum rearranged) ；out_prev = (C @ prev) * sd
    out_prev = []
    for bi in range(b):
        per_c = []
        for ci in range(c):
            per_l = []
            for li in range(lc):
                per_h = []
                for hi in range(h):
                    # state_decay_out[b,c,l,h,1] = exp(dA_cumsum[b,h,c,l])
                    sd = math.exp(dA_cumsum[bi][hi][ci][li])
                    row = []
                    for pi in range(p):
                        acc = 0.0
                        for ni in range(n):
                            cv = C_exp[bi][ci * lc + li][hi][ni]
                            acc += cv * prev_states[bi][ci][hi][pi][ni]
                        row.append(acc * sd)
                    per_h.append(row)
                per_l.append(per_h)
            per_c.append(per_l)
        out_prev.append(per_c)

    # ---- 合并并还原布局
    out = []
    for bi in range(b):
        per_l = []
        for li in range(l):
            ci = li // lc
            sub = li % lc
            per_h = []
            for hi in range(h):
                row = []
                for pi in range(p):
                    row.append(out_intra[bi][ci][sub][hi][pi] + out_prev[bi][ci][sub][hi][pi])
                per_h.append(row)
            per_l.append(per_h)
        out.append(per_l)

    # ---- D 跳连
    if D is not None:
        # 官方：if D.dim() == 1: D = rearrange(D, "h -> h 1")
        #      ⇒ 一维 [nheads] 变成 [nheads, 1]；二维 [nheads, headdim] 原样使用
        if len(D) and isinstance(D[0], (list, tuple)):
            if len(D) != h or len(D[0]) != p:
                raise ValueError("D must be [nheads] or [nheads][headdim]")
            for bi in range(b):
                for li in range(l):
                    for hi in range(h):
                        for pi in range(p):
                            out[bi][li][hi][pi] += x[bi][li][hi][pi] * float(D[hi][pi])
        else:
            if len(D) != h:
                raise ValueError("D must be [nheads] or [nheads][headdim]")
            for bi in range(b):
                for li in range(l):
                    for hi in range(h):
                        for pi in range(p):
                            out[bi][li][hi][pi] += x[bi][li][hi][pi] * float(D[hi])

    # ---- z 门控
    if z is not None:
        for bi in range(b):
            for li in range(l):
                for hi in range(h):
                    for pi in range(p):
                        out[bi][li][hi][pi] *= _silu(float(z[bi][li][hi][pi]))

    return {
        "CB": CB,
        "decay": decay,
        "scores_decay": scores_decay,
        "out": out,
    }


# ---------------------------------------------- mode 4: ssd_forward 全编排
def ssd_forward(x, dt, A, B_mat, C_mat, chunk_size, D=None, z=None,
                dt_bias=None, dt_softplus=False):
    """ssd_chunk_scan_combined_ref 的等价实现（三段编排）。"""
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    b = len(x)
    l = len(x[0])
    h = len(x[0][0])
    p = len(x[0][0][0])
    g = len(B_mat[0][0])
    n = len(B_mat[0][0][0])
    if h % g != 0:
        raise ValueError("nheads must be divisible by ngroups")
    _check_shape(x, (b, l, h, p), "x")
    _check_shape(dt, (b, l, h), "dt")
    _check_shape(B_mat, (b, l, g, n), "B_mat")
    _check_shape(C_mat, (b, l, g, n), "C_mat")
    _check_flat(A, h, "A")
    if dt_bias is not None:
        _check_flat(dt_bias, h, "dt_bias")

    # ---- 官方 ssd_chunk_scan_combined_ref 在 seqlen % chunk_size != 0 时会崩：
    #      它把 dt 补齐到 chunk 整数倍，但 chunk_scan_ref 里那句
    #      `assert seqlen == nchunks * chunk_size` 用的是【原始 seqlen】⇒ 矛盾。
    #      本题要求在这一情形显式报错（而不是静默补齐）。
    if l % chunk_size != 0:
        raise ValueError("ssd_forward: seqlen must be divisible by chunk_size "
                         "(official ssd_chunk_scan_combined_ref asserts this)")
    pad = 0
    dt_pad = []
    for bi in range(b):
        rows = [[float(dt[bi][li][hi]) for hi in range(h)] for li in range(l)]
        for _ in range(pad):
            rows.append([0.0] * h)
        dt_pad.append(rows)
    l_tot = l + pad
    c = l_tot // chunk_size
    lc = chunk_size

    # ---- 官方：dt = rearrange(dt, "b (c l) h -> b h c l", l=chunk_size)
    dt4 = []
    for bi in range(b):
        per_h = []
        for hi in range(h):
            per_c = []
            for ci in range(c):
                per_c.append([float(dt_pad[bi][ci * lc + li][hi]) for li in range(lc)])
            per_h.append(per_c)
        dt4.append(per_h)

    # ---- 官方：dt = dt.float()（本题已是 double，跳过）
    # ---- 官方：if dt_bias is not None: dt = dt + rearrange(dt_bias, "h -> h 1 1")
    if dt_bias is not None:
        if len(dt_bias) != h:
            raise ValueError("dt_bias must have nheads elements")
        for bi in range(b):
            for hi in range(h):
                for ci in range(c):
                    for li in range(lc):
                        dt4[bi][hi][ci][li] += float(dt_bias[hi])

    # ---- 官方：if dt_softplus: dt = F.softplus(dt)
    if dt_softplus:
        for bi in range(b):
            for hi in range(h):
                for ci in range(c):
                    for li in range(lc):
                        v = dt4[bi][hi][ci][li]
                        dt4[bi][hi][ci][li] = math.log1p(math.exp(v)) if v > 30.0 else \
                            (v if v < -30.0 else math.log1p(math.exp(v)))

    # ---- 第一段：chunk_state
    cs = chunk_state(x, dt4, A, B_mat, chunk_size)
    dA_cumsum = cs["dA_cumsum"]
    states = cs["states"]

    # ---- 第二段：state_passing
    # 官方：rearrange(states, "... p n -> ... (p n)")
    flat_states = []
    for bi in range(b):
        per_c = []
        for ci in range(c):
            per_h = []
            for hi in range(h):
                row = []
                for pi in range(p):
                    for ni in range(n):
                        row.append(states[bi][ci][hi][pi][ni])
                per_h.append(row)
            per_c.append(per_h)
        flat_states.append(per_c)

    dA_chunk_cumsum = []
    for bi in range(b):
        per_h = []
        for hi in range(h):
            per_h.append([dA_cumsum[bi][hi][ci][lc - 1] for ci in range(c)])
        dA_chunk_cumsum.append(per_h)

    sp = state_passing(flat_states, dA_chunk_cumsum)

    # 官方：rearrange(passed, "... (p n) -> ... p n", n=dstate)
    passed = sp["passed"]
    prev_states = []
    for bi in range(b):
        per_c = []
        for ci in range(c):
            per_h = []
            for hi in range(h):
                per_p = []
                for pi in range(p):
                    per_n = []
                    for ni in range(n):
                        per_n.append(passed[bi][ci][hi][pi * n + ni])
                    per_p.append(per_n)
                per_h.append(per_p)
            per_c.append(per_h)
        prev_states.append(per_c)

    # ---- 第三段：chunk_scan
    scan = chunk_scan(B_mat, C_mat, x, dt4, dA_cumsum, prev_states, D=D, z=z)

    return {
        "out": scan["out"],
        "dA_cumsum": dA_cumsum,
        "decay_states": cs["decay_states"],
        "states": states,
        "prev_states": prev_states,
        "decay_chunk": sp["decay_chunk"],
        "final_state": sp["final_state"],
        "CB": scan["CB"],
        "scores_decay": scan["scores_decay"],
        "nchunks": c,
        "chunk_size": lc,
        "padded_seqlen": l_tot,
    }


# ----------------------------------------------------------------- 统一入口
def solve(mode, **kw):
    if mode == "chunk_state":
        return chunk_state(kw["x"], kw["dt"], kw["A"], kw["B_mat"], kw["chunk_size"])
    if mode == "state_passing":
        return state_passing(kw["states"], kw["dA_chunk_cumsum"],
                             kw.get("initial_states"))
    if mode == "chunk_scan":
        return chunk_scan(kw["B_mat"], kw["C_mat"], kw["x"], kw["dt"],
                          kw["dA_cumsum"], kw["prev_states"],
                          D=kw.get("D"), z=kw.get("z"))
    if mode == "ssd_forward":
        return ssd_forward(kw["x"], kw["dt"], kw["A"], kw["B_mat"], kw["C_mat"],
                           kw["chunk_size"], D=kw.get("D"), z=kw.get("z"),
                           dt_bias=kw.get("dt_bias"),
                           dt_softplus=kw.get("dt_softplus", False))
    raise ValueError("unknown mode: %r" % (mode,))
