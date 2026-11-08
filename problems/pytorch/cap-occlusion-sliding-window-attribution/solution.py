"""Captum Occlusion（滑动窗口扰动归因）的纯 Python 数值账。

对应官方源码：
  captum/attr/_core/occlusion.py::Occlusion.attribute
  captum/attr/_core/occlusion.py::Occlusion._occlusion_mask
  captum/attr/_core/feature_ablation.py::FeatureAblation._process_ablated_out_full
  captum/attr/_core/feature_ablation.py::format_result

四个 mode：
  occlusion_grid        —— 网格枚举账（shift_counts / 窗口数 / 每窗偏移 / 截断后的覆盖区间）
  occlusion_mask        —— 掩码与覆盖计数账（每窗 0/1 掩码 / 每特征被覆盖次数）
  occlusion_diff        —— 差分账（初始输出 / 每窗遮挡后输出 / eval_diff / 未归一化累加）
  occlusion_attribution —— 完整归因账（未归一化累加 / 权重 / 归一化后的归因）

所有数值走 fractions.Fraction 精确有理数，最终输出才转 float。
"""

import math
import struct
from fractions import Fraction


# --------------------------------------------------------------------------
# 基础工具
# --------------------------------------------------------------------------


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


def _as_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("%s must be an integer, got %r" % (name, value))
    return value


def _flatten(nested):
    """把嵌套列表展平成一维列表（只支持数值叶子）。"""
    if not isinstance(nested, (list, tuple)):
        return [nested]
    out = []
    for item in nested:
        out.extend(_flatten(item))
    return out


def _prod(values):
    out = 1
    for v in values:
        out *= v
    return out


def _reshape(flat, shape):
    """把一维列表按 shape 还原成嵌套列表。"""
    if not shape:
        return flat[0]
    if len(shape) == 1:
        return list(flat)
    step = _prod(shape[1:])
    return [_reshape(flat[i * step:(i + 1) * step], shape[1:]) for i in range(shape[0])]


def _f32(value):
    """把一个 Python float 舍入到 float32（IEEE-754 单精度最近偶数舍入）。"""
    try:
        return struct.unpack("f", struct.pack("f", value))[0]
    except (OverflowError, struct.error):
        return float("inf") if value > 0 else float("-inf")


def _frac_to_float(value):
    """Fraction -> float；分母为 0 时按官方语义给 nan/inf。"""
    try:
        return float(value)
    except (OverflowError, ZeroDivisionError):
        return float("nan") if value.denominator == 0 else float("inf")


# --------------------------------------------------------------------------
# 参数校验（对应官方 _format_and_verify_sliding_window_shapes /
# _format_and_verify_strides / Occlusion.attribute 里的两条 assert）
# --------------------------------------------------------------------------


def _normalize_shape(shape):
    if not isinstance(shape, (list, tuple)) or len(shape) == 0:
        raise ValueError("shape must be a non-empty list of positive integers")
    dims = []
    for d in shape:
        d = _as_int(d, "shape entry")
        if d <= 0:
            raise ValueError("shape entries must be positive, got %r" % (d,))
        dims.append(d)
    return dims


def _normalize_window(window_shape, dims):
    if not isinstance(window_shape, (list, tuple)) or len(window_shape) != len(dims):
        raise ValueError(
            "sliding_window_shapes must have one entry per non-batch dimension "
            "(expected %d, got %r)" % (len(dims), window_shape)
        )
    out = []
    for w in window_shape:
        w = _as_int(w, "sliding_window_shapes entry")
        if w <= 0:
            raise ValueError("sliding window dimensions must be positive, got %r" % (w,))
        out.append(w)
    return out


def _normalize_strides(strides, dims, window_shape):
    """strides 允许 None（默认全 1）。

    官方对应两条约束：
      1) 窗口维度不得超过输入维度；
      2) 某维若窗口未恰好覆盖该维（current_shape != 0），则该维 stride 不得大于窗口长度。
    """
    if strides is None:
        return [1] * len(dims)
    # 官方 _format_and_verify_strides：strides 可以是一个标量 int
    # （对单输入张量会被包成 (strides,)），此时该标量适用于所有维度。
    if isinstance(strides, int) and not isinstance(strides, bool):
        strides = [strides] * len(dims)
    if not isinstance(strides, (list, tuple)) or len(strides) != len(dims):
        raise ValueError(
            "strides must have one entry per non-batch dimension "
            "(expected %d, got %r)" % (len(dims), strides)
        )
    out = []
    for s in strides:
        s = _as_int(s, "strides entry")
        if s <= 0:
            raise ValueError("strides must be positive, got %r" % (s,))
        out.append(s)
    for i in range(len(dims)):
        if window_shape[i] > dims[i]:
            raise ValueError(
                "Sliding window dimensions %r cannot exceed input dimensions %r."
                % (tuple(window_shape), tuple(dims))
            )
        current_shape = dims[i] - window_shape[i]
        if current_shape != 0 and out[i] > window_shape[i]:
            raise ValueError(
                "Stride dimension %d cannot be larger than sliding window "
                "shape dimension %d." % (out[i], window_shape[i])
            )
    return out


def _shift_counts(dims, window_shape, strides):
    """官方 shift_counts：ceil((dim - window) / stride) + 1。"""
    counts = []
    for i in range(len(dims)):
        current_shape = dims[i] - window_shape[i]
        counts.append(int(math.ceil(current_shape / strides[i])) + 1)
    return counts


def _enumerate_offsets(shift_counts, strides):
    """base-conversion 枚举：dim0 是最低位（变化最快）。

    官方 _occlusion_mask：
      for i, shift_count in enumerate(shift_counts):
          current_index.append((remaining_total % shift_count) * stride)
          remaining_total = remaining_total // shift_count
    """
    total = _prod(shift_counts)
    offsets = []
    for idx in range(total):
        remaining = idx
        cur = []
        for i, sc in enumerate(shift_counts):
            cur.append((remaining % sc) * strides[i])
            remaining //= sc
        offsets.append(cur)
    return offsets


def _window_extent(offset, window_shape, dims):
    """窗口在每维的实际覆盖区间 [start, stop)，末尾可能被截断。

    官方用 torch.nn.functional.pad 做这件事：右侧 padding 为负数时
    F.pad 会把窗口尾部裁掉，等价于把 stop 压到 dims[i]。
    """
    out = []
    for i in range(len(dims)):
        start = offset[i]
        stop = min(start + window_shape[i], dims[i])
        out.append([start, stop])
    return out


def _build_mask(offset, window_shape, dims):
    """构造某窗口的 0/1 掩码（展平，长度 prod(dims)）。

    窗口的截断只有 _window_extent 一个真源：官方用 torch.nn.functional.pad
    的负 padding 实现「窗口末端被裁掉」，等价于把该维的覆盖区间右端
    压到 dims[i]。这里直接复用 _window_extent 的结果，不再各写一份截断逻辑。
    """
    total = _prod(dims)
    mask = [0] * total
    ext = _window_extent(offset, window_shape, dims)
    if len(dims) == 1:
        for c in range(ext[0][0], ext[0][1]):
            mask[c] = 1
        return mask
    # 多维：按行优先（row-major）展开
    strides = [_prod(dims[i + 1:]) for i in range(len(dims))]
    for idx in range(total):
        rem = idx
        coords = []
        for st in strides:
            coords.append(rem // st)
            rem = rem % st
        ok = True
        for i in range(len(dims)):
            if not (ext[i][0] <= coords[i] < ext[i][1]):
                ok = False
                break
        if ok:
            mask[idx] = 1
    return mask


# --------------------------------------------------------------------------
# mode 1: 网格枚举账
# --------------------------------------------------------------------------


def occlusion_grid(shape, window_shape, strides=None):
    dims = _normalize_shape(shape)
    win = _normalize_window(window_shape, dims)
    st = _normalize_strides(strides, dims, win)
    counts = _shift_counts(dims, win, st)
    offsets = _enumerate_offsets(counts, st)
    extents = [_window_extent(off, win, dims) for off in offsets]
    return {
        "shift_counts": counts,
        "num_windows": len(offsets),
        "offsets": offsets,
        "extents": extents,
    }


# --------------------------------------------------------------------------
# mode 2: 掩码与覆盖计数账
# --------------------------------------------------------------------------


def occlusion_mask(shape, window_shape, strides=None):
    dims = _normalize_shape(shape)
    win = _normalize_window(window_shape, dims)
    st = _normalize_strides(strides, dims, win)
    counts = _shift_counts(dims, win, st)
    offsets = _enumerate_offsets(counts, st)
    masks = [_build_mask(off, win, dims) for off in offsets]
    coverage = [0] * _prod(dims)
    for m in masks:
        for i, v in enumerate(m):
            if v:
                coverage[i] += 1
    return {
        "masks": masks,
        "coverage": coverage,
    }


# --------------------------------------------------------------------------
# 前向：显式 MLP（层间 ReLU，最后一层不加 ReLU）
# --------------------------------------------------------------------------


def _normalize_layers(layers):
    if not isinstance(layers, (list, tuple)) or len(layers) == 0:
        raise ValueError("layers must be a non-empty list")
    out = []
    for layer in layers:
        if not isinstance(layer, dict):
            raise ValueError("each layer must be a dict with 'W' and 'b'")
        w = layer.get("W")
        b = layer.get("b")
        if not isinstance(w, (list, tuple)) or len(w) == 0:
            raise ValueError("layer['W'] must be a non-empty list of rows")
        rows = []
        for row in w:
            if not isinstance(row, (list, tuple)):
                raise ValueError("layer['W'] rows must be lists")
            rows.append([_to_frac(v) for v in row])
        n_out = len(rows)
        n_in = len(rows[0])
        for row in rows:
            if len(row) != n_in:
                raise ValueError("layer['W'] rows must all have the same length")
        if b is None:
            bias = [Fraction(0)] * n_out
        else:
            if not isinstance(b, (list, tuple)) or len(b) != n_out:
                raise ValueError("layer['b'] must have one entry per output row")
            bias = [_to_frac(v) for v in b]
        out.append((rows, bias))
    for i in range(1, len(out)):
        prev_out = len(out[i - 1][0])
        cur_in = len(out[i][0][0])
        if prev_out != cur_in:
            raise ValueError(
                "layer %d expects input dim %d but previous layer outputs %d"
                % (i, cur_in, prev_out)
            )
    return out


def _forward(layers, x):
    """x 为长度 prod(dims) 的有理数列表；返回最后一层输出的有理数列表。"""
    h = list(x)
    n_layers = len(layers)
    for li, (rows, bias) in enumerate(layers):
        n_in = len(rows[0])
        if len(h) != n_in:
            raise ValueError(
                "forward input length %d does not match layer %d input dim %d"
                % (len(h), li, n_in)
            )
        out = []
        for r in range(len(rows)):
            row = rows[r]
            acc = bias[r]
            for c in range(n_in):
                acc = acc + row[c] * h[c]
            out.append(acc)
        if li != n_layers - 1:
            out = [v if v > 0 else Fraction(0) for v in out]
        h = out
    return h


def _normalize_baseline(baseline, dims):
    """baseline 支持标量，或与 shape 同形的嵌套列表 / 等长一维列表。"""
    total = _prod(dims)
    if isinstance(baseline, (list, tuple)):
        flat = _flatten(baseline)
        if len(flat) != total:
            raise ValueError(
                "baseline length %d does not match input size %d" % (len(flat), total)
            )
        return [_to_frac(v) for v in flat]
    return [_to_frac(baseline)] * total


def _normalize_target(target, n_outputs):
    """归一化 target 索引。

    官方走 _verify_select_column：`output[(slice(None), *target)]`，
    target 是 int 时被包成 `(target,)`，于是走的是 **PyTorch 的负索引语义**
    —— target=-1 选最后一个输出，target=-n 选第一个。
    越界（target >= n 或 target < -n）由 PyTorch 抛 IndexError。

    这里保留同样的语义：负索引折算成 n + target，越界则报错。
    """
    if isinstance(target, bool) or not isinstance(target, int):
        raise ValueError("target must be an integer index")
    if target < 0:
        if target < -n_outputs:
            raise ValueError(
                "target index %d is out of range for %d outputs" % (target, n_outputs)
            )
        return target + n_outputs
    if target >= n_outputs:
        raise ValueError(
            "target index %d is out of range for %d outputs" % (target, n_outputs)
        )
    return target


def _occlusion_core(layers, x, baseline, shape, window_shape, strides, target):
    """四个 mode 共用的核心计算。"""
    dims = _normalize_shape(shape)
    win = _normalize_window(window_shape, dims)
    st = _normalize_strides(strides, dims, win)
    total = _prod(dims)

    flat_x = _flatten(x)
    if len(flat_x) != total:
        raise ValueError(
            "input length %d does not match shape size %d" % (len(flat_x), total)
        )
    xf = [_to_frac(v) for v in flat_x]

    norm_layers = _normalize_layers(layers)
    base = _normalize_baseline(baseline, dims)

    counts = _shift_counts(dims, win, st)
    offsets = _enumerate_offsets(counts, st)

    # 先跑一次前向拿到输出维数，才能校验 target
    probe = _forward(norm_layers, xf)
    tgt = _normalize_target(target, len(probe))
    initial_eval = probe[tgt]

    window_evals = []
    eval_diffs = []
    total_attrib = [Fraction(0)] * total
    weights = [0] * total

    for off in offsets:
        mask = _build_mask(off, win, dims)
        ablated = [
            xf[i] * (1 - mask[i]) + base[i] * mask[i] for i in range(total)
        ]
        out = _forward(norm_layers, ablated)[tgt]
        window_evals.append(out)
        diff = initial_eval - out
        eval_diffs.append(diff)
        for i in range(total):
            if mask[i]:
                total_attrib[i] = total_attrib[i] + diff
                weights[i] += 1

    return {
        "dims": dims,
        "window_shape": win,
        "strides": st,
        "shift_counts": counts,
        "offsets": offsets,
        "initial_eval": initial_eval,
        "window_evals": window_evals,
        "eval_diffs": eval_diffs,
        "total_attrib": total_attrib,
        "weights": weights,
    }


# --------------------------------------------------------------------------
# mode 3: 差分账
# --------------------------------------------------------------------------


def occlusion_diff(layers, x, baseline, shape, window_shape, strides, target):
    core = _occlusion_core(
        layers, x, baseline, shape, window_shape, strides, target
    )
    return {
        "initial_eval": _frac_to_float(core["initial_eval"]),
        "window_evals": [_frac_to_float(v) for v in core["window_evals"]],
        "eval_diffs": [_frac_to_float(v) for v in core["eval_diffs"]],
        "total_attrib": [_frac_to_float(v) for v in core["total_attrib"]],
    }


# --------------------------------------------------------------------------
# mode 4: 完整归因账
# --------------------------------------------------------------------------


def occlusion_attribution(layers, x, baseline, shape, window_shape, strides, target):
    core = _occlusion_core(
        layers, x, baseline, shape, window_shape, strides, target
    )
    total_attrib = core["total_attrib"]
    weights = core["weights"]
    attribution = []
    for i in range(len(total_attrib)):
        if weights[i] == 0:
            # 官方是 0/0 -> nan；这里显式给 nan
            attribution.append(float("nan"))
        else:
            # 官方 format_result 里是 `single_attrib.float() / weight`：
            # 先把 float64 的累加值降级到 float32，再在 float32 下做除法。
            attribution.append(_f32(_f32(_frac_to_float(total_attrib[i]))
                                    / _f32(weights[i])))
    return {
        "total_attrib": [_frac_to_float(v) for v in total_attrib],
        "weights": list(weights),
        "attribution": attribution,
    }


# --------------------------------------------------------------------------
# 单入口
# --------------------------------------------------------------------------


def solve(mode, **kw):
    if mode == "occlusion_grid":
        return occlusion_grid(
            kw["shape"], kw["window_shape"], kw.get("strides")
        )
    if mode == "occlusion_mask":
        return occlusion_mask(
            kw["shape"], kw["window_shape"], kw.get("strides")
        )
    if mode == "occlusion_diff":
        return occlusion_diff(
            kw["layers"], kw["x"], kw["baseline"], kw["shape"],
            kw["window_shape"], kw.get("strides"), kw["target"],
        )
    if mode == "occlusion_attribution":
        return occlusion_attribution(
            kw["layers"], kw["x"], kw["baseline"], kw["shape"],
            kw["window_shape"], kw.get("strides"), kw["target"],
        )
    raise ValueError("unknown mode: %r" % (mode,))
