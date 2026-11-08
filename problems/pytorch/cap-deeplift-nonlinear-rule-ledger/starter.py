"""Captum DeepLIFT 非线性传播规则账 —— 骨架。

你需要实现四个 mode，由 solve(mode, **kw) 分派：

  chunk_diffs(inputs, outputs)
      → {"delta_in": [...], "delta_out": [...], "half": int}

  nonlinear_rule(inputs, outputs, grad_input, grad_output, eps=1e-10)
      → {"delta_in", "delta_out", "new_grad_in", "used_fallback", "n_fallback"}

  softmax_rule(inputs, outputs, grad_input, grad_output, eps=1e-10)
      → {"unnorm", "total", "n", "shift", "new_grad_in"}

  maxpool_rule(inputs, outputs, grad_input, grad_output, indices, eps=1e-10)
      → {"delta_in", "delta_out_xmax", "delta_out", "scaled",
         "unpool_delta", "unpool_ref", "merged", "new_grad_in", "used_fallback"}

关键口径：

- 所有入参都是展平的一维列表；inputs / outputs 长度必须为偶数。
- chunk(2) 沿第 0 维拆分：前半 = 实际输入（actual），后半 = 参考输入（reference）。
- grad_input 与 inputs 同长；grad_output 与 outputs 同长。
- indices 长度 = len(outputs)，前半是 actual 侧的窗口命中位置，
  后半是 reference 侧的窗口命中位置。
- 所有非法输入一律抛 ValueError。
- 能走精确有理数的部分走 fractions.Fraction；最终所有出参统一转 float。

四个容易写错的点：

1. _compute_diffs 返回的是 cat(2 * [delta]) —— 差分被【复制成两份】，
   所以返回长度等于【入参长度】，不是入参的一半。
2. nonlinear 的判据是 abs(delta_in) < eps（【严格小于】），
   恰好等于 eps 时走【除法支】；默认 eps 是 1e-10。
3. softmax 的 n 是 grad_input.numel()（【全部元素个数】），
   减的是【一个全局标量】 sum * 1/n，不是逐行减均值。
4. maxpool 的 unpool 分配在【同一位置被多次命中】时是【后写覆盖】，
   不是累加；eps 分支取 grad_input[0]（【第一半】，不是第 0 个元素）。

请在 solve 里完成实现，不要修改 test_cases.py。
"""

import math
from fractions import Fraction

_EPS = 1e-10


def _to_frac(value):
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
    if not isinstance(value, (list, tuple)):
        return [value]
    out = []
    for item in value:
        out.extend(_flat(item))
    return out


def _norm_list(value, name):
    if not isinstance(value, (list, tuple)):
        raise ValueError("%s must be a list" % name)
    return [_to_frac(v) for v in _flat(value)]


def _f(value):
    return float(value)


def _check_same_len(lists, names):
    lens = [len(x) for x in lists]
    if len(set(lens)) != 1:
        raise ValueError(
            "length mismatch: %s" % ", ".join(
                "%s=%d" % (n, l) for n, l in zip(names, lens))
        )


def _check_even(value, name):
    if value % 2 != 0:
        raise ValueError("%s length %d must be even (chunk(2))" % (name, value))


# --------------------------------------------------------------------------
# TODO: 实现下面四个函数
# --------------------------------------------------------------------------


def chunk_diffs(inputs, outputs):
    """_compute_diffs 的 chunk(2) + cat(2 * [...]) 语义。"""
    raise NotImplementedError("TODO: chunk_diffs")


def nonlinear_rule(inputs, outputs, grad_input, grad_output, eps=_EPS):
    """nonlinear 的 where 分支。"""
    raise NotImplementedError("TODO: nonlinear_rule")


def softmax_rule(inputs, outputs, grad_input, grad_output, eps=_EPS):
    """softmax 的全局归一化修正。"""
    raise NotImplementedError("TODO: softmax_rule")


def maxpool_rule(inputs, outputs, grad_input, grad_output, indices, eps=_EPS):
    """maxpool 的 cross-maximum 拆分与 unpool 分配。"""
    raise NotImplementedError("TODO: maxpool_rule")


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


# --------------------------------------------------------------------------
# 自检
# --------------------------------------------------------------------------


if __name__ == "__main__":
    print("skeleton loaded — implement the four modes above.")
