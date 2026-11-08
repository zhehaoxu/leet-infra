"""Captum DeepLIFT 非线性传播规则账（纯 Python）。

对应官方源码（pytorch/captum, main 分支）：
  captum/attr/_core/deep_lift.py::_compute_diffs
  captum/attr/_core/deep_lift.py::nonlinear
  captum/attr/_core/deep_lift.py::softmax
  captum/attr/_core/deep_lift.py::maxpool

四个 mode（全部显式接收长度，不依赖任何隐式状态）：
  chunk_diffs      —— _compute_diffs 的 chunk(2) 语义与 cat(2*[...]) 复制
  nonlinear_rule   —— nonlinear 的 where 分支（严格小于 eps 走 grad_input）
  softmax_rule     —— softmax 的全局归一化修正（减 total / n）
  maxpool_rule     —— maxpool 的 cross-maximum delta 拆分与 unpool 分配

数值口径：能走精确有理数（Fraction）的部分一律走 Fraction；
最终所有出参统一转成 float。
"""

import math
from fractions import Fraction


# --------------------------------------------------------------------------
# 基础工具
# --------------------------------------------------------------------------

_EPS = 1e-10


def _to_frac(value):
    """把输入转成精确有理数 Fraction。"""
    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):
        raise TypeError("bool is not a valid numeric value")
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        return Fraction(value)
    if isinstance(value, str):
        return Fraction(value)
    raise TypeError("unsupported numeric type: %r" % (type(value),))


def _flat(value):
    """把嵌套列表展平成一维列表。"""
    if not isinstance(value, (list, tuple)):
        return [value]
    out = []
    for item in value:
        out.extend(_flat(item))
    return out


def _norm_list(value, name):
    """把入参归一化成一维 Fraction 列表。"""
    if not isinstance(value, (list, tuple)):
        raise ValueError("%s must be a list" % name)
    return [_to_frac(v) for v in _flat(value)]


def _f(value):
    """出参统一转 float。"""
    if isinstance(value, Fraction):
        return float(value)
    return float(value)


def _check_same_len(lists, names):
    """校验所有列表等长。"""
    lens = [len(x) for x in lists]
    if len(set(lens)) != 1:
        raise ValueError(
            "length mismatch: %s" % ", ".join(
                "%s=%d" % (n, l) for n, l in zip(names, lens))
        )


def _check_even(value, name):
    """chunk(2) 要求长度为偶数。"""
    if value % 2 != 0:
        raise ValueError("%s length %d must be even (chunk(2))" % (name, value))


# --------------------------------------------------------------------------
# mode 1: chunk_diffs
# --------------------------------------------------------------------------


def chunk_diffs(inputs, outputs):
    """_compute_diffs 的 chunk(2) 语义。

    官方实现：
        input, input_ref = inputs.chunk(2)
        output, output_ref = outputs.chunk(2)
        delta_in = input - input_ref
        delta_out = output - output_ref
        return torch.cat(2 * [delta_in]), torch.cat(2 * [delta_out])

    ★ 要点：
      1. chunk(2) 沿 dim0 把张量拆成【前半 = 实际输入】与【后半 = reference】；
      2. 返回值是 cat(2 * [...]) —— 每份差分被【复制成两份】，
         所以返回长度等于入参长度（不是入参的一半）；
      3. 前一半与后一半内容完全相同。

    返回：
      {"delta_in": [...], "delta_out": [...], "half": int}
    """
    xs = _norm_list(inputs, "inputs")
    ys = _norm_list(outputs, "outputs")
    _check_same_len([xs, ys], ["inputs", "outputs"])
    if not xs:
        raise ValueError("inputs must not be empty")
    _check_even(len(xs), "inputs")
    _check_even(len(ys), "outputs")

    half = len(xs) // 2
    input_ = xs[:half]
    input_ref = xs[half:]
    output_ = ys[:half]
    output_ref = ys[half:]

    delta_in = [input_[i] - input_ref[i] for i in range(half)]
    delta_out = [output_[i] - output_ref[i] for i in range(half)]

    # 官方是 torch.cat(2 * [delta]) —— 复制两份
    return {
        "delta_in": [delta_in, delta_in],
        "delta_out": [delta_out, delta_out],
        "half": half,
    }


# --------------------------------------------------------------------------
# mode 2: nonlinear_rule
# --------------------------------------------------------------------------


def nonlinear_rule(inputs, outputs, grad_input, grad_output, eps=_EPS):
    """nonlinear 的 where 分支。

    官方实现：
        delta_in, delta_out = _compute_diffs(inputs, outputs)
        new_grad_inp = torch.where(
            abs(delta_in) < eps, grad_input, grad_output * delta_out / delta_in
        )

    ★ 要点：
      1. 判据是【严格小于】abs(delta_in) < eps —— 恰等于 eps 时走除法支；
      2. 走除法支时是 grad_output * delta_out / delta_in（先乘后除）；
      3. delta_in 为负时结果符号随之为负；
      4. grad_input / grad_output 的长度必须与 delta_in 相同
         （即与 inputs 同长，因为 _compute_diffs 复制了两份）。

    返回：
      {"delta_in", "delta_out", "new_grad_in", "used_fallback", "n_fallback"}
    """
    d = chunk_diffs(inputs, outputs)
    delta_in = d["delta_in"][0] + d["delta_in"][1]
    delta_out = d["delta_out"][0] + d["delta_out"][1]

    gi = _norm_list(grad_input, "grad_input")
    go = _norm_list(grad_output, "grad_output")
    _check_same_len(
        [gi, go, delta_in], ["grad_input", "grad_output", "delta_in"])

    eps_f = _to_frac(eps)
    new_grad = []
    used_fallback = []
    for i in range(len(delta_in)):
        if abs(delta_in[i]) < eps_f:
            new_grad.append(gi[i])
            used_fallback.append(True)
        else:
            new_grad.append(go[i] * delta_out[i] / delta_in[i])
            used_fallback.append(False)

    n_fb = sum(1 for b in used_fallback if b)
    return {
        "delta_in": delta_in,
        "delta_out": delta_out,
        "new_grad_in": new_grad,
        "used_fallback": used_fallback,
        "n_fallback": n_fb,
    }


# --------------------------------------------------------------------------
# mode 3: softmax_rule
# --------------------------------------------------------------------------


def softmax_rule(inputs, outputs, grad_input, grad_output, eps=_EPS):
    """softmax 的归一化修正。

    官方实现：
        delta_in, delta_out = _compute_diffs(inputs, outputs)
        grad_input_unnorm = torch.where(
            abs(delta_in) < eps, grad_input, grad_output * delta_out / delta_in
        )
        # normalizing
        n = grad_input.numel()
        new_grad_inp = grad_input_unnorm - grad_input_unnorm.sum() * 1 / n

    ★ 要点：
      1. 先算未归一化的版本（与 nonlinear 完全一样）；
      2. n = grad_input.numel() —— 【全部元素个数】，不是 batch 数，
         也不是 delta 的「单侧长度」；
      3. 减去的是 grad_input_unnorm.sum() * 1/n —— 一个【全局标量】，
         不是逐行（逐 batch）减均值；
      4. 修正后所有元素之和为 0。

    返回：
      {"unnorm", "total", "n", "shift", "new_grad_in"}
    """
    r = nonlinear_rule(inputs, outputs, grad_input, grad_output, eps)
    unnorm = r["new_grad_in"]
    total = Fraction(0)
    for v in unnorm:
        total += v
    n = len(unnorm)
    shift = total * Fraction(1, n)
    out = [v - shift for v in unnorm]
    return {
        "unnorm": unnorm,
        "total": total,
        "n": n,
        "shift": shift,
        "new_grad_in": out,
    }


# --------------------------------------------------------------------------
# mode 4: maxpool_rule
# --------------------------------------------------------------------------


def maxpool_rule(inputs, outputs, grad_input, grad_output, indices, eps=_EPS):
    """maxpool 的 cross-maximum delta 拆分与 unpool 分配。

    官方实现（captum/attr/_core/deep_lift.py::maxpool）：
        input, input_ref = inputs.chunk(2)
        output, output_ref = outputs.chunk(2)
        delta_in = input - input_ref
        delta_in = torch.cat(2 * [delta_in])
        delta_out_xmax = torch.max(output, output_ref)
        delta_out = torch.cat([delta_out_xmax - output_ref, output - delta_out_xmax])

        _, indices = pool_func(module.input, module.kernel_size, ...)
        grad_output_updated = grad_output
        unpool_grad_out_delta, unpool_grad_out_ref_delta = torch.chunk(
            unpool_func(
                grad_output_updated * delta_out,
                indices,
                module.kernel_size, module.stride, module.padding,
                list(cast(torch.Size, module.input.shape)),
            ),
            2,
        )
        unpool_grad_out_delta = unpool_grad_out_delta + unpool_grad_out_ref_delta
        unpool_grad_out_delta = torch.cat(2 * [unpool_grad_out_delta])

        new_grad_inp = torch.where(
            abs(delta_in) < eps, grad_input[0], unpool_grad_out_delta / delta_in
        )

    ★ 要点：
      1. delta_out_xmax = max(output, output_ref) —— 【逐元素取两半的最大值】；
      2. delta_out = cat([xmax - output_ref, output - xmax])
         —— 前半是「xmax 相对 reference 的超出量」，
            后半是「实际输出相对 xmax 的缺口」（可为负）；
      3. grad_output 与 outputs 【同形】（长度 = 2 * n_out），
         delta_out 与 grad_output 是【逐元素】相乘，不是广播；
      4. unpool 把梯度按 indices scatter 回输入位置：
         前半用 actual 的窗口命中位置（配 delta_out 前半），
         后半用 reference 的窗口命中位置（配 delta_out 后半）；
      5. ★ scatter 到同一位置时是【后写覆盖】（不是累加）；
      6. 两半 scatter 结果【相加】后再 cat(2*[...]) 复制两份；
      7. eps 分支取的是 grad_input[0] —— 【第一半】(沿 dim0)，
         不是 grad_input 的第 0 个元素。

    返回：
      {"delta_in", "delta_out_xmax", "delta_out", "scaled", "unpool_delta",
       "unpool_ref", "merged", "new_grad_in", "used_fallback"}
    """
    xs = _norm_list(inputs, "inputs")
    ys = _norm_list(outputs, "outputs")
    gi = _norm_list(grad_input, "grad_input")
    go = _norm_list(grad_output, "grad_output")
    idx = _norm_list(indices, "indices")

    if not xs:
        raise ValueError("inputs must not be empty")
    _check_even(len(xs), "inputs")
    _check_even(len(ys), "outputs")

    n_feat = len(xs) // 2      # 单侧（actual 或 reference）的输入展平长度
    n_out = len(ys) // 2       # 单侧的输出展平长度

    input_ = xs[:n_feat]
    input_ref = xs[n_feat:]
    output_ = ys[:n_out]
    output_ref = ys[n_out:]

    if len(gi) != len(xs):
        raise ValueError(
            "grad_input length %d does not match inputs length %d"
            % (len(gi), len(xs))
        )
    # grad_output 与 outputs 同形（长度 = 2 * n_out），
    # delta_out 也是 2 * n_out，两者逐元素相乘
    if len(go) != 2 * n_out:
        raise ValueError(
            "grad_output length %d does not match 2 * n_out = %d"
            % (len(go), 2 * n_out)
        )
    if len(idx) != 2 * n_out:
        raise ValueError(
            "indices length %d does not match 2 * n_out = %d"
            % (len(idx), 2 * n_out)
        )

    # delta_in：单侧差分后复制两份（官方 cat(2 * [delta_in])）
    delta_in_half = [input_[i] - input_ref[i] for i in range(n_feat)]
    delta_in = delta_in_half + delta_in_half

    # cross-maximum：逐元素取 actual 与 reference 的最大值
    delta_out_xmax = []
    for i in range(n_out):
        a = output_[i]
        b = output_ref[i]
        delta_out_xmax.append(a if a > b else b)

    # delta_out = cat([xmax - output_ref, output - xmax])
    delta_out = (
        [delta_out_xmax[i] - output_ref[i] for i in range(n_out)]
        + [output_[i] - delta_out_xmax[i] for i in range(n_out)]
    )

    # grad_output_updated * delta_out —— 逐元素（两者同长）
    scaled = [go[i] * delta_out[i] for i in range(2 * n_out)]

    # unpool：按 indices 把梯度 scatter 回输入位置
    # ★ 官方走 torch.max_unpool，同址冲突时是【后写覆盖】
    unpool_delta = [Fraction(0)] * n_feat
    unpool_ref = [Fraction(0)] * n_feat
    for w in range(n_out):
        p = int(idx[w])
        unpool_delta[p] = scaled[w]
        p2 = int(idx[n_out + w])
        unpool_ref[p2] = scaled[n_out + w]

    # 两半相加后 cat(2*[...]) 复制两份
    merged_half = [unpool_delta[i] + unpool_ref[i] for i in range(n_feat)]
    merged = merged_half + merged_half

    eps_f = _to_frac(eps)
    new_grad = []
    used_fallback = []
    for i in range(len(delta_in)):
        if abs(delta_in[i]) < eps_f:
            # ★ grad_input[0] —— 第一半（沿 dim0），两半都用它
            new_grad.append(gi[i] if i < n_feat else gi[i - n_feat])
            used_fallback.append(True)
        else:
            new_grad.append(merged[i] / delta_in[i])
            used_fallback.append(False)

    return {
        "delta_in": delta_in,
        "delta_out_xmax": delta_out_xmax,
        "delta_out": delta_out,
        "scaled": scaled,
        "unpool_delta": unpool_delta,
        "unpool_ref": unpool_ref,
        "merged": merged,
        "new_grad_in": new_grad,
        "used_fallback": used_fallback,
    }


# --------------------------------------------------------------------------
# 单入口
# --------------------------------------------------------------------------


def solve(mode, **kw):
    if mode == "chunk_diffs":
        return chunk_diffs(kw["inputs"], kw["outputs"])
    if mode == "nonlinear_rule":
        return nonlinear_rule(
            kw["inputs"], kw["outputs"], kw["grad_input"], kw["grad_output"],
            kw.get("eps", _EPS),
        )
    if mode == "softmax_rule":
        return softmax_rule(
            kw["inputs"], kw["outputs"], kw["grad_input"], kw["grad_output"],
            kw.get("eps", _EPS),
        )
    if mode == "maxpool_rule":
        return maxpool_rule(
            kw["inputs"], kw["outputs"], kw["grad_input"], kw["grad_output"],
            kw["indices"], kw.get("eps", _EPS),
        )
    raise ValueError("unknown mode: %r" % (mode,))
