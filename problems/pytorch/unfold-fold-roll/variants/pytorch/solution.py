"""参考答案：unfold / fold / roll 三合一。

实现思路：
- unfold：先把输入零填充到 (H+2ph, W+2pw)，再对窗口内每个核偏移 (i, j) 做
  一次跨步切片（N, C, H_out, W_out），写入 cols 视图 (N, C, kh*kw, L) 的
  第 i*kw+j 行——核元素只有常数个，主计算全部是向量化切片拷贝；
- fold：反向操作，把 cols 视图按 (i, j) 切片后散射累加（+=）回带 padding 的
  输出画布，最后裁掉 padding，重叠区自然求和；
- roll：逐维用 (arange(n) - shift) mod n 的索引重排（等价于输出位置 i 取
  输入位置 (i - shift) mod n，即 (i + shift) mod n 的循环移位）。
"""

import torch


def _pair(v, name):
    """int 或长度 2 的序列 -> (h, w) 二元组。"""
    if isinstance(v, bool):
        raise ValueError(f"{name} 必须是 int 或长度为 2 的元组，got {v!r}")
    if isinstance(v, int):
        return (v, v)
    if isinstance(v, (tuple, list)) and len(v) == 2:
        return (int(v[0]), int(v[1]))
    raise ValueError(f"{name} 必须是 int 或长度为 2 的元组，got {v!r}")


def _out_size(size, k, s, d, p):
    return (size + 2 * p - d * (k - 1) - 1) // s + 1


def _unfold(x, kernel_size, stride, dilation, padding):
    n, c, h, w = x.shape
    kh, kw = kernel_size
    sh, sw = stride
    dh, dw = dilation
    ph, pw = padding
    h_out = _out_size(h, kh, sh, dh, ph)
    w_out = _out_size(w, kw, sw, dw, pw)
    if h_out <= 0 or w_out <= 0:
        raise RuntimeError(
            f"Given input with spatial size ({h}, {w}), kernel_size={kernel_size}, "
            f"dilation={dilation}, padding={padding}, calculated shape of the array "
            f"of sliding blocks as ({h_out}, {w_out}), which is too small")
    xp = torch.zeros(n, c, h + 2 * ph, w + 2 * pw, dtype=x.dtype)
    xp[:, :, ph:ph + h, pw:pw + w] = x
    cols = torch.zeros(n, c * kh * kw, h_out * w_out, dtype=x.dtype)
    view = cols.view(n, c, kh * kw, h_out * w_out)
    for i in range(kh):
        for j in range(kw):
            patch = xp[:, :, i * dh: i * dh + sh * h_out: sh,
                       j * dw: j * dw + sw * w_out: sw]
            view[:, :, i * kw + j, :] = patch.reshape(n, c, h_out * w_out)
    return cols


def _fold(cols, output_size, kernel_size, stride, dilation, padding):
    if cols.dim() != 3:
        raise ValueError(
            f"mode='fold' 要求输入为 (N, C*kh*kw, L) 三维张量，got shape {tuple(cols.shape)}")
    if output_size is None:
        raise ValueError("mode='fold' 时必须提供 output_size=(H, W)")
    oh, ow = _pair(output_size, "output_size")
    n, ckk, l = cols.shape
    kh, kw = kernel_size
    if ckk % (kh * kw) != 0:
        raise RuntimeError(
            f"Expected size of input's dimension 1 to be divisible by the product "
            f"of kernel_size, but got input.size(1)={ckk} and kernel_size={kernel_size}")
    c = ckk // (kh * kw)
    sh, sw = stride
    dh, dw = dilation
    ph, pw = padding
    h_out = _out_size(oh, kh, sh, dh, ph)
    w_out = _out_size(ow, kw, sw, dw, pw)
    if l != h_out * w_out:
        raise RuntimeError(
            f"Given output_size=({oh}, {ow}), kernel_size={kernel_size}, "
            f"dilation={dilation}, padding={padding}, stride={stride}, expected size "
            f"of input's dimension 2 to match the calculated number of sliding blocks "
            f"{h_out} * {w_out} = {h_out * w_out}, but got input.size(2)={l}")
    buf = torch.zeros(n, c, oh + 2 * ph, ow + 2 * pw, dtype=cols.dtype)
    view = cols.view(n, c, kh * kw, l)
    for i in range(kh):
        for j in range(kw):
            patch = view[:, :, i * kw + j, :].reshape(n, c, h_out, w_out)
            buf[:, :, i * dh: i * dh + sh * h_out: sh,
                j * dw: j * dw + sw * w_out: sw] += patch
    return buf[:, :, ph:ph + oh, pw:pw + ow].contiguous()


def _roll(x, shifts, dims):
    ndim = x.dim()
    if isinstance(shifts, (tuple, list)):
        if dims is None:
            raise ValueError("shifts 为元组时必须提供等长的 dims 元组")
        if not isinstance(dims, (tuple, list)):
            raise ValueError("shifts 为元组时 dims 必须是等长元组")
        shift_list = [int(s) for s in shifts]
        dim_list = [int(v) for v in dims]
        if len(shift_list) != len(dim_list):
            raise ValueError(
                f"shifts 和 dims 长度必须一致: {len(shift_list)} vs {len(dim_list)}")
    else:
        s = int(shifts)
        if dims is None:
            dim_list = [ndim - 1]  # 本题约定：dims 缺省对最后一维 roll
        elif isinstance(dims, (tuple, list)):
            dim_list = [int(dims[0])]  # 本题约定：shifts 为 int 时对 dims[0] roll
        else:
            dim_list = [int(dims)]
        shift_list = [s]
    out = x
    for s, d in zip(shift_list, dim_list):
        d = d + ndim if d < 0 else d
        if not 0 <= d < ndim:
            raise IndexError(
                f"Dimension out of range (expected to be in range of "
                f"[{-ndim}, {ndim - 1}], but got {d})")
        n = out.shape[d]
        if n > 0:
            idx = (torch.arange(n) - s) % n
            out = out.index_select(d, idx)
    return out


def solve(x, kernel_size, stride=None, dilation=1, padding=0,
          mode="unfold", output_size=None, shifts=None, dims=None):
    if mode == "unfold":
        if x.dim() != 4:
            raise ValueError(
                f"mode='unfold' 要求 x 为 (N, C, H, W) 四维张量，got {x.dim()} 维")
        return _unfold(x, _pair(kernel_size, "kernel_size"),
                       _pair(stride if stride is not None else kernel_size, "stride"),
                       _pair(dilation, "dilation"), _pair(padding, "padding"))
    if mode == "fold":
        return _fold(x, output_size, _pair(kernel_size, "kernel_size"),
                     _pair(stride if stride is not None else kernel_size, "stride"),
                     _pair(dilation, "dilation"), _pair(padding, "padding"))
    if mode == "roll":
        if shifts is None:
            raise ValueError("mode='roll' 时必须提供 shifts")
        return _roll(x, shifts, dims)
    raise ValueError(f"unknown mode: {mode!r}（合法取值 unfold/fold/roll）")
