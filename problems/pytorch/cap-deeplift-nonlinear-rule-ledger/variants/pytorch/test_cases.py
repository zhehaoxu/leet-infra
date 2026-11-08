"""847 判题用例：Captum DeepLIFT 非线性传播规则账。

四个 check_* 函数，均以 solve 作为唯一参数（judge 内部以 fn(solve) 调用）：
  check_chunk_diffs      —— _compute_diffs 的 chunk(2) + cat(2*[...]) 复制
  check_nonlinear_rule   —— nonlinear 的 where 分支（严格小于 eps）
  check_softmax_rule     —— softmax 的全局归一化修正
  check_maxpool_rule     —— maxpool 的 cross-maximum 拆分 + unpool 分配

每个 check 都带一份【独立 oracle】（不依赖参考解，按官方算式重写一遍）。
"""

import math


# --------------------------------------------------------------------------
# 通用工具
# --------------------------------------------------------------------------


def _flat(value):
    if not isinstance(value, (list, tuple)):
        return [value]
    out = []
    for item in value:
        out.extend(_flat(item))
    return out


def _f(value):
    return [float(v) for v in _flat(value)]


def _close(a, b, tol=1e-9):
    a = _f(a)
    b = _f(b)
    if len(a) != len(b):
        return False, "length mismatch: %d vs %d" % (len(a), len(b))
    worst = 0.0
    for i, (x, y) in enumerate(zip(a, b)):
        d = abs(float(x) - float(y))
        if d > worst:
            worst = d
    if worst > tol:
        return False, "max abs diff %.3e at index %d" % (worst, i)
    return True, ""


def _get(res, key):
    """兼容 solve 返回 dict 或对象属性两种风格。"""
    if isinstance(res, dict):
        if key not in res:
            raise AssertionError("missing key: %s" % key)
        return res[key]
    if not hasattr(res, key):
        raise AssertionError("missing attribute: %s" % key)
    return getattr(res, key)


# --------------------------------------------------------------------------
# 独立 oracle
# --------------------------------------------------------------------------


def _oracle_chunk(inputs, outputs):
    h = len(inputs) // 2
    di = [inputs[i] - inputs[i + h] for i in range(h)]
    do = [outputs[i] - outputs[i + h] for i in range(h)]
    return di + di, do + do


def _oracle_nonlinear(inputs, outputs, gi, go, eps):
    di, do = _oracle_chunk(inputs, outputs)
    out = []
    for i in range(len(di)):
        if abs(di[i]) < eps:
            out.append(gi[i])
        else:
            out.append(go[i] * do[i] / di[i])
    return out


def _oracle_softmax(inputs, outputs, gi, go, eps):
    un = _oracle_nonlinear(inputs, outputs, gi, go, eps)
    total = 0.0
    for v in un:
        total += v
    n = len(un)
    return [v - total / n for v in un]


def _oracle_maxpool(inputs, outputs, gi, go, indices):
    n_feat = len(inputs) // 2
    n_out = len(outputs) // 2
    di_half = [inputs[i] - inputs[i + n_feat] for i in range(n_feat)]
    di = di_half + di_half
    o_act = outputs[:n_out]
    o_ref = outputs[n_out:]
    xmax = [max(o_act[i], o_ref[i]) for i in range(n_out)]
    dout = ([xmax[i] - o_ref[i] for i in range(n_out)]
            + [o_act[i] - xmax[i] for i in range(n_out)])
    scaled = [go[i] * dout[i] for i in range(2 * n_out)]
    up_d = [0.0] * n_feat
    up_r = [0.0] * n_feat
    for w in range(n_out):
        # 后写覆盖（不是累加）
        up_d[int(indices[w])] = scaled[w]
        up_r[int(indices[n_out + w])] = scaled[n_out + w]
    merged_half = [up_d[i] + up_r[i] for i in range(n_feat)]
    merged = merged_half + merged_half
    out = []
    for i in range(len(di)):
        if abs(di[i]) < 1e-10:
            out.append(gi[i] if i < n_feat else gi[i - n_feat])
        else:
            out.append(merged[i] / di[i])
    return out


# --------------------------------------------------------------------------
# mode 1: chunk_diffs
# --------------------------------------------------------------------------


def check_chunk_diffs(solve):
    """_compute_diffs：chunk(2) 拆分 + cat(2*[...]) 复制两份。"""

    # 基础：前半 actual、后半 reference
    inp = [1.0, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5]
    out = [10.0, 20.0, 30.0, 40.0, 5.0, 15.0, 25.0, 35.0]
    r = solve("chunk_diffs", inputs=inp, outputs=out)
    di = _f(_get(r, "delta_in"))
    do = _f(_get(r, "delta_out"))
    exp_di, exp_do = _oracle_chunk(inp, out)
    ok, msg = _close(di, exp_di)
    assert ok, "delta_in mismatch: %s" % msg
    ok, msg = _close(do, exp_do)
    assert ok, "delta_out mismatch: %s" % msg

    # ★ 关键：长度必须是【入参长度】而不是入参的一半，且两半内容相同
    assert len(di) == len(inp), (
        "delta_in length %d must equal inputs length %d "
        "(cat(2 * [delta]) duplicates)" % (len(di), len(inp)))
    assert len(do) == len(out), (
        "delta_out length %d must equal outputs length %d" % (len(do), len(out)))
    h = len(inp) // 2
    assert di[:h] == di[h:], "delta_in halves must be identical"
    assert do[:h] == do[h:], "delta_out halves must be identical"

    # 差分方向：actual - reference（不是 reference - actual）
    assert abs(di[0] - (1.0 - 0.5)) < 1e-12, "delta_in must be actual - reference"
    assert abs(do[0] - (10.0 - 5.0)) < 1e-12, "delta_out must be actual - reference"

    # 负差分
    inp2 = [1.0, 2.0, 3.0, 4.0, 3.0, 3.0, 3.0, 3.0]
    out2 = [10.0, 20.0, 30.0, 40.0, 1.0, 2.0, 3.0, 4.0]
    r2 = solve("chunk_diffs", inputs=inp2, outputs=out2)
    di2 = _f(_get(r2, "delta_in"))
    exp_di2, _ = _oracle_chunk(inp2, out2)
    ok, msg = _close(di2, exp_di2)
    assert ok, "negative delta_in mismatch: %s" % msg
    assert di2[0] < 0, "delta_in must keep the sign of (actual - reference)"

    # 更长的非对称数据
    inp3 = [2.0, -1.0, 0.5, 3.0, 7.0, -2.5, 1.0, 0.0,
            4.0, 4.0, 9.0, 1.0, 0.5, 0.5, 2.0, 6.0]
    out3 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
            0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
    r3 = solve("chunk_diffs", inputs=inp3, outputs=out3)
    di3 = _f(_get(r3, "delta_in"))
    do3 = _f(_get(r3, "delta_out"))
    exp_di3, exp_do3 = _oracle_chunk(inp3, out3)
    ok, msg = _close(di3, exp_di3)
    assert ok, "long asymmetric delta_in mismatch: %s" % msg
    ok, msg = _close(do3, exp_do3)
    assert ok, "long asymmetric delta_out mismatch: %s" % msg
    assert len(di3) == 16 and len(do3) == 16


# --------------------------------------------------------------------------
# mode 2: nonlinear_rule
# --------------------------------------------------------------------------


def check_nonlinear_rule(solve):
    """nonlinear：where(abs(delta_in) < eps, grad_input, go*do/di)。"""

    inp = [1.0, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5]
    out = [10.0, 20.0, 30.0, 40.0, 5.0, 15.0, 25.0, 35.0]
    gi = [1.0, 2.0, 3.0, 4.0, 9.0, 9.0, 9.0, 9.0]
    go = [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]

    r = solve("nonlinear_rule", inputs=inp, outputs=out,
              grad_input=gi, grad_output=go, eps=1e-10)
    y = _f(_get(r, "new_grad_in"))
    exp = _oracle_nonlinear(inp, out, gi, go, 1e-10)
    ok, msg = _close(y, exp)
    assert ok, "nonlinear mismatch: %s" % msg

    # ★ 长度翻倍：输出长度 = inputs 长度
    assert len(y) == len(inp), (
        "new_grad_in length %d must equal inputs length %d" % (len(y), len(inp)))

    # ★ eps 分支：delta_in 恰为 0 时取 grad_input
    inz = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]
    rz = solve("nonlinear_rule", inputs=inz, outputs=out,
               grad_input=gi, grad_output=go, eps=1e-10)
    yz = _f(_get(rz, "new_grad_in"))
    assert abs(yz[0] - gi[0]) < 1e-12, (
        "delta_in == 0 must take grad_input branch, got %r" % yz[0])
    assert abs(yz[4] - gi[4]) < 1e-12, (
        "delta_in == 0 must take grad_input branch, got %r" % yz[4])

    # ★ eps 边界：判据是【严格小于】
    # |delta_in| == 0.5：eps=0.5 -> 走除法支；eps=0.5000001 -> 走 grad_input 支
    ine = [1.0, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5]   # delta_in = 0.5
    r_eq = solve("nonlinear_rule", inputs=ine, outputs=out,
                 grad_input=gi, grad_output=go, eps=0.5)
    y_eq = _f(_get(r_eq, "new_grad_in"))
    exp_eq = _oracle_nonlinear(ine, out, gi, go, 0.5)
    ok, msg = _close(y_eq, exp_eq)
    assert ok, "eps boundary (==) mismatch: %s" % msg
    # 走除法支时结果 != grad_input
    assert abs(y_eq[0] - gi[0]) > 1e-12, (
        "|delta_in| == eps must take the division branch (strict <)")

    r_gt = solve("nonlinear_rule", inputs=ine, outputs=out,
                 grad_input=gi, grad_output=go, eps=0.5000001)
    y_gt = _f(_get(r_gt, "new_grad_in"))
    assert abs(y_gt[0] - gi[0]) < 1e-12, (
        "|delta_in| < eps must take the grad_input branch")
    assert abs(y_gt[4] - gi[4]) < 1e-12, (
        "grad_input branch must use the matching index")

    r_lt = solve("nonlinear_rule", inputs=ine, outputs=out,
                 grad_input=gi, grad_output=go, eps=0.4999999)
    y_lt = _f(_get(r_lt, "new_grad_in"))
    exp_lt = _oracle_nonlinear(ine, out, gi, go, 0.4999999)
    ok, msg = _close(y_lt, exp_lt)
    assert ok, "eps boundary (<) mismatch: %s" % msg

    # ★ 负 delta_in：结果符号随之为负
    inn = [1.0, 2.0, 3.0, 4.0, 3.0, 3.0, 3.0, 3.0]    # delta_in = [-2,-1,0,1]
    rn = solve("nonlinear_rule", inputs=inn, outputs=out,
               grad_input=gi, grad_output=go, eps=1e-10)
    yn = _f(_get(rn, "new_grad_in"))
    exp_n = _oracle_nonlinear(inn, out, gi, go, 1e-10)
    ok, msg = _close(yn, exp_n)
    assert ok, "negative delta mismatch: %s" % msg
    assert yn[0] < 0, "negative delta_in must give a negative result"

    # ★ delta_out 为 0 -> 结果为 0
    outz = [10.0, 20.0, 10.0, 20.0, 10.0, 20.0, 10.0, 20.0]
    rz2 = solve("nonlinear_rule", inputs=inp, outputs=outz,
                grad_input=gi, grad_output=go, eps=1e-10)
    yz2 = _f(_get(rz2, "new_grad_in"))
    exp_z2 = _oracle_nonlinear(inp, outz, gi, go, 1e-10)
    ok, msg = _close(yz2, exp_z2)
    assert ok, "zero delta_out mismatch: %s" % msg

    # 更长的非对称数据
    inb = [2.0, -1.0, 0.5, 3.0, 7.0, -2.5, 1.0, 0.0,
           4.0, 4.0, 9.0, 1.0, 0.5, 0.5, 2.0, 6.0]
    outb = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
            0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
    gib = [0.5, -1.5, 2.0, 0.0, 3.0, -2.0, 1.0, 4.0,
           1.0, 1.0, 2.0, 2.0, -3.0, 0.5, 0.0, 6.0]
    gob = [2.0, 1.0, -1.0, 3.0, 0.5, 2.5, -2.0, 1.0,
           1.5, -0.5, 2.0, 0.0, 4.0, -3.0, 1.0, 2.0]
    rb = solve("nonlinear_rule", inputs=inb, outputs=outb,
               grad_input=gib, grad_output=gob, eps=1e-10)
    yb = _f(_get(rb, "new_grad_in"))
    exp_b = _oracle_nonlinear(inb, outb, gib, gob, 1e-10)
    ok, msg = _close(yb, exp_b)
    assert ok, "long asymmetric mismatch: %s" % msg
    assert len(yb) == 16

    # ★ 默认 eps：官方 nonlinear 的默认 eps 是 1e-10。
    #   这里用 delta_in = 1e-8 —— 它落在 1e-10 与 1e-6 之间，
    #   于是「默认 1e-10 -> 走除法支」与「默认 1e-6 -> 走 grad_input 支」
    #   的两种实现会给出不同结果，从而把默认值钉死。
    ind = [1.0, 2.0, 3.0, 4.0, 1.0 - 1e-8, 2.0, 3.0, 4.0]
    rd = solve("nonlinear_rule", inputs=ind, outputs=out,
               grad_input=gi, grad_output=go)
    yd = _f(_get(rd, "new_grad_in"))
    exp_d = _oracle_nonlinear(ind, out, gi, go, 1e-10)
    ok, msg = _close(yd, exp_d)
    assert ok, "default eps mismatch: %s" % msg
    assert abs(yd[4] - gi[4]) > 1e-12, (
        "default eps must be 1e-10 (delta_in=1e-8 must take the "
        "division branch), got %r" % yd[4])


# --------------------------------------------------------------------------
# mode 3: softmax_rule
# --------------------------------------------------------------------------


def check_softmax_rule(solve):
    """softmax：全局归一化修正（减 total / numel），不是逐行减均值。"""

    inp = [1.0, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5]
    out = [10.0, 20.0, 30.0, 40.0, 5.0, 15.0, 25.0, 35.0]
    gi = [1.0, 2.0, 3.0, 4.0, 9.0, 9.0, 9.0, 9.0]
    go = [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]

    r = solve("softmax_rule", inputs=inp, outputs=out,
              grad_input=gi, grad_output=go, eps=1e-10)
    y = _f(_get(r, "new_grad_in"))
    exp = _oracle_softmax(inp, out, gi, go, 1e-10)
    ok, msg = _close(y, exp)
    assert ok, "softmax mismatch: %s" % msg

    # ★ 修正后所有元素之和为 0（全局标量修正的性质）
    total = 0.0
    for v in y:
        total += v
    assert abs(total) < 1e-9, (
        "softmax output must sum to 0 (global scalar shift), got %r" % total)

    # ★ n 是【全部元素个数】= inputs 长度（因为差分被复制成两份）
    #    逐行减均值与全局减标量在【不对称】数据上结果不同
    # 用 delta_in=0.5（使前半 0.5 / 后半 -0.5 的 unnorm 呈对称）会掩盖差异，
    # 所以这里刻意选不对称的数据。
    inb = [2.0, -1.0, 0.5, 3.0, 7.0, -2.5, 1.0, 0.0,
           4.0, 4.0, 9.0, 1.0, 0.5, 0.5, 2.0, 6.0]
    outb = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
            0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
    gib = [0.5, -1.5, 2.0, 0.0, 3.0, -2.0, 1.0, 4.0,
           1.0, 1.0, 2.0, 2.0, -3.0, 0.5, 0.0, 6.0]
    gob = [2.0, 1.0, -1.0, 3.0, 0.5, 2.5, -2.0, 1.0,
           1.5, -0.5, 2.0, 0.0, 4.0, -3.0, 1.0, 2.0]
    rb = solve("softmax_rule", inputs=inb, outputs=outb,
               grad_input=gib, grad_output=gob, eps=1e-10)
    yb = _f(_get(rb, "new_grad_in"))
    exp_b = _oracle_softmax(inb, outb, gib, gob, 1e-10)
    ok, msg = _close(yb, exp_b)
    assert ok, "long asymmetric softmax mismatch: %s" % msg

    # ★ 逐行（逐 half）减均值是【另一种】实现，必须与全局标量修正区分开。
    #   这里刻意选不对称数据，使两种实现给出不同结果（否则本例无法区分）。
    h = len(inb) // 2
    un = _oracle_nonlinear(inb, outb, gib, gob, 1e-10)
    row_mean = []
    for half in (un[:h], un[h:]):
        m = sum(half) / len(half)
        row_mean.extend([v - m for v in half])
    global_shift = sum(un) / len(un)
    global_ver = [v - global_shift for v in un]
    # 两种候选必须不同，否则本例丧失区分能力
    assert max(abs(a - b) for a, b in zip(row_mean, global_ver)) > 1e-6, (
        "test data must make the per-row-mean and global-scalar variants "
        "differ")
    # 而 solve 的结果必须落在【全局标量】那一侧
    assert max(abs(a - b) for a, b in zip(row_mean, yb)) > 1e-6, (
        "softmax must subtract a GLOBAL scalar (sum/numel), "
        "not a per-row mean")

    # ★ unnorm / n 必须一起暴露，便于核对「减的是哪个标量」
    un_norm = _f(_get(r, "unnorm"))
    exp_un = _oracle_nonlinear(inp, out, gi, go, 1e-10)
    ok, msg = _close(un_norm, exp_un)
    assert ok, "softmax unnorm mismatch: %s" % msg
    n = _get(r, "n")
    assert int(n) == len(inp), (
        "n must be grad_input.numel() = %d, got %r" % (len(inp), n))
    shift = _f(_get(r, "shift"))
    exp_shift = sum(exp_un) / len(exp_un)
    ok, msg = _close(shift, [exp_shift])
    assert ok, "softmax shift mismatch: %s" % msg

    # 全部 delta_in 为 0 的极端情形
    inz = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    rz = solve("softmax_rule", inputs=inz, outputs=out,
               grad_input=gi, grad_output=go, eps=1e-10)
    yz = _f(_get(rz, "new_grad_in"))
    exp_z = _oracle_softmax(inz, out, gi, go, 1e-10)
    ok, msg = _close(yz, exp_z)
    assert ok, "all-zero delta_in softmax mismatch: %s" % msg


# --------------------------------------------------------------------------
# mode 4: maxpool_rule
# --------------------------------------------------------------------------


def check_maxpool_rule(solve):
    """maxpool：cross-maximum 拆分 + unpool 分配（后写覆盖）。"""

    # ---- case 1: 基础 ----
    inp = [1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0]
    out = [2.0, 4.0, 0.0, 0.0]
    gi = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]
    go = [1.0, 2.0, 1.0, 2.0]
    idx = [1, 3, 0, 2]
    r = solve("maxpool_rule", inputs=inp, outputs=out,
              grad_input=gi, grad_output=go, indices=idx)
    y = _f(_get(r, "new_grad_in"))
    exp = _oracle_maxpool(inp, out, gi, go, idx)
    ok, msg = _close(y, exp)
    assert ok, "maxpool mismatch: %s" % msg
    assert len(y) == len(inp), (
        "new_grad_in length %d must equal inputs length %d" % (len(y), len(inp)))

    # ★ cross-maximum：xmax 逐元素取两半最大值
    xmax = _f(_get(r, "delta_out_xmax"))
    assert len(xmax) == len(out) // 2, "delta_out_xmax length must be n_out"
    assert abs(xmax[0] - 2.0) < 1e-12, "xmax must be elementwise max"
    assert abs(xmax[1] - 4.0) < 1e-12, "xmax must be elementwise max"

    # ★ delta_out 的前半 = xmax - output_ref，后半 = output - xmax
    dout = _f(_get(r, "delta_out"))
    assert len(dout) == 2 * (len(out) // 2), "delta_out length must be 2*n_out"
    assert abs(dout[0] - (2.0 - 0.0)) < 1e-12, (
        "delta_out[:n_out] must be xmax - output_ref")
    assert abs(dout[2] - (2.0 - 2.0)) < 1e-12, (
        "delta_out[n_out:] must be output - xmax")

    # ---- case 2: reference 侧输出更大 -> xmax 取 reference ----
    inp2 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    out2 = [2.0, 4.0, 6.0, 8.0]
    gi2 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    go2 = [1.0, 1.0, 2.0, 2.0]
    idx2 = [1, 3, 1, 3]
    r2 = solve("maxpool_rule", inputs=inp2, outputs=out2,
               grad_input=gi2, grad_output=go2, indices=idx2)
    y2 = _f(_get(r2, "new_grad_in"))
    exp2 = _oracle_maxpool(inp2, out2, gi2, go2, idx2)
    ok, msg = _close(y2, exp2)
    assert ok, "maxpool (ref larger) mismatch: %s" % msg
    xmax2 = _f(_get(r2, "delta_out_xmax"))
    assert abs(xmax2[0] - 6.0) < 1e-12, (
        "xmax must take the reference side when it is larger")
    dout2 = _f(_get(r2, "delta_out"))
    # reference 侧更大 -> 前半为 0（xmax==output_ref），后半为负
    assert abs(dout2[0] - 0.0) < 1e-12, (
        "when reference is larger, delta_out[:n_out] must be 0")
    assert dout2[2] < 0, (
        "when reference is larger, delta_out[n_out:] must be negative")

    # ---- case 3: stride < kernel -> 同址冲突 ----
    inp3 = [1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    out3 = [3.0, 4.0, 5.0, 0.0, 0.0, 0.0]
    gi3 = [1.0] * 10
    go3 = [2.0, 3.0, 4.0, 0.0, 0.0, 0.0]
    idx3 = [2, 3, 4, 0, 1, 2]
    r3 = solve("maxpool_rule", inputs=inp3, outputs=out3,
               grad_input=gi3, grad_output=go3, indices=idx3)
    y3 = _f(_get(r3, "new_grad_in"))
    exp3 = _oracle_maxpool(inp3, out3, gi3, go3, idx3)
    ok, msg = _close(y3, exp3)
    assert ok, "maxpool (collision) mismatch: %s" % msg

    # ---- case 4: eps 分支 -> 取 grad_input[0]（第一半）----
    inpe = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]
    oute = [2.0, 4.0, 2.0, 4.0]
    gie = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]
    goe = [1.0, 2.0, 1.0, 2.0]
    idxe = [1, 3, 1, 3]
    re = solve("maxpool_rule", inputs=inpe, outputs=oute,
               grad_input=gie, grad_output=goe, indices=idxe)
    ye = _f(_get(re, "new_grad_in"))
    expe = _oracle_maxpool(inpe, oute, gie, goe, idxe)
    ok, msg = _close(ye, expe)
    assert ok, "maxpool (eps branch) mismatch: %s" % msg
    # delta_in 全为 0 -> 取 grad_input[0]（第一半），两半都用它
    assert abs(ye[0] - 7.0) < 1e-12, (
        "eps branch must take grad_input[0] (first half)")
    assert abs(ye[4] - 7.0) < 1e-12, (
        "eps branch must take grad_input[0] (first half) for BOTH halves")

    # ---- case 5: 同址冲突下必须是【后写覆盖】而不是累加 ----
    inpc = [1.0, 9.0, 0.0, 0.0, 0.0, 0.0,
            2.0, 8.0, 0.0, 0.0, 0.0, 0.0]
    outc = [9.0, 9.0, 0.0, 0.0, 8.0, 8.0, 0.0, 0.0]
    gic = [1.0] * 12
    goc = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
    idxc = [1, 1, 2, 3, 1, 1, 2, 3]
    rc = solve("maxpool_rule", inputs=inpc, outputs=outc,
               grad_input=gic, grad_output=goc, indices=idxc)
    yc = _f(_get(rc, "new_grad_in"))
    expc = _oracle_maxpool(inpc, outc, gic, goc, idxc)
    ok, msg = _close(yc, expc)
    assert ok, "maxpool (overwrite conflict) mismatch: %s" % msg
    merged = _f(_get(rc, "merged"))
    # 位置1 被写了两次（scaled 10 与 20）：覆盖 -> 20；累加 -> 30
    assert abs(merged[1] - 20.0) < 1e-12, (
        "unpool must OVERWRITE on index collision (torch.max_unpool "
        "semantics), got %r" % merged[1])

    # ---- case 6: 负 delta ----
    inpn = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    outn = [2.0, 4.0, 6.0, 1.0, 3.0, 5.0]
    gin = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
           7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    gon = [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
    idxn = [1, 3, 5, 1, 3, 5]
    rn = solve("maxpool_rule", inputs=inpn, outputs=outn,
               grad_input=gin, grad_output=gon, indices=idxn)
    yn = _f(_get(rn, "new_grad_in"))
    expn = _oracle_maxpool(inpn, outn, gin, gon, idxn)
    ok, msg = _close(yn, expn)
    assert ok, "maxpool (negative delta) mismatch: %s" % msg
