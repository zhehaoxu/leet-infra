"""参考答案：LpPool（1-D/2-D）+ maxpool indices + maxunpool —— pad/unfold/scatter 实现。

口径说明：lppool 按 (Σ_窗口 |x_i|^p)^(1/p) 计算（带 |x|，对负输入的奇数 / 分数 p 也良定义）；
p=2（默认）时与 F.lp_pool2d 逐位一致（原生实现对负输入的 p=1 是带符号求和、分数 p 会 NaN）。
maxpool_idx 的 indices 是最大值在原输入（未填充）空间图上的行优先展平偏移，与
F.max_pool2d(return_indices=True) 完全一致（含 padding 偏移修正、并列取行优先最前者）。
"""

import math

import torch
import torch.nn.functional as F


def _pair2(v, what):
    """int → (v, v)；长度 1/2 的整数序列 → 二元组；其余抛 ValueError。"""
    if isinstance(v, int) and not isinstance(v, bool):
        return (v, v)
    if isinstance(v, (tuple, list)) and len(v) in (1, 2) and all(
            isinstance(d, int) and not isinstance(d, bool) for d in v):
        return (v[0], v[0]) if len(v) == 1 else (v[0], v[1])
    raise ValueError(f"{what} 必须是 int 或长度 1/2 的整数序列，got {v!r}")


def _check_p(p):
    pf = float(p)
    if not math.isfinite(pf) or pf < 1.0:
        raise ValueError(f"p 必须 >= 1 且有限，got {p!r}")
    return pf


def _unpool_out_size(small, k, s, output_size):
    h_s, w_s = small
    if output_size is None:
        # 按无重叠池化的几何关系推断
        return (h_s - 1) * s[0] + k[0], (w_s - 1) * s[1] + k[1]
    oh, ow = _pair2(output_size, "output_size")
    if oh < h_s or ow < w_s:
        raise ValueError(f"output_size {(oh, ow)} 须逐维 >= 小图尺寸 {(h_s, w_s)}")
    return oh, ow


def solve(x, kernel_size, stride=None, p=2, mode="lppool2d", padding=0,
          output_size=None, indices=None):
    if mode not in ("lppool2d", "lppool1d", "maxpool_idx", "maxunpool"):
        raise ValueError(f"unknown mode: {mode!r}")
    kh, kw = _pair2(kernel_size, "kernel_size")
    sh, sw = _pair2(stride, "stride") if stride is not None else (kh, kw)
    ph, pw = _pair2(padding, "padding")
    for dims, name in (((kh, kw), "kernel_size"), ((sh, sw), "stride")):
        if any(d < 1 for d in dims):
            raise ValueError(f"{name} 各维必须是正整数，got {dims}")
    if ph < 0 or pw < 0:
        raise ValueError(f"padding 必须非负，got {(ph, pw)}")

    if mode == "lppool1d":
        pf = _check_p(p)
        if x.dim() != 3:
            raise ValueError(f"lppool1d 要求 x 恰 3 维 (N,C,L)，got {tuple(x.shape)}")
        if kh != kw or sh != sw or ph != pw:
            raise ValueError("lppool1d 的 kernel/stride/padding 逐维必须一致"
                             "（传 int 或单元素元组）")
        if 2 * ph > kh:
            raise ValueError(f"padding 每边须 <= kernel/2，got padding={ph}, kernel={kh}")
        l_in = x.shape[-1]
        ol = (l_in + 2 * ph - kh) // sh + 1
        if ol < 1:
            raise ValueError(f"窗口放不进输入：L={l_in}, k={kh}, s={sh}, padding={ph}")
        xp = F.pad(x, (ph, ph), value=0.0)
        win = xp.unfold(2, kh, sh)  # (N, C, ol, k)
        return win.abs().pow(pf).sum(dim=-1).pow(1.0 / pf)

    if mode == "lppool2d":
        pf = _check_p(p)
        if x.dim() != 4:
            raise ValueError(f"lppool2d 要求 x 恰 4 维 (N,C,H,W)，got {tuple(x.shape)}")
        if 2 * ph > kh or 2 * pw > kw:
            raise ValueError(f"padding 每边须 <= kernel/2，got padding={(ph, pw)}, "
                             f"kernel={(kh, kw)}")
        n, c, h, w = x.shape
        oh = (h + 2 * ph - kh) // sh + 1
        ow = (w + 2 * pw - kw) // sw + 1
        if oh < 1 or ow < 1:
            raise ValueError(f"窗口放不进输入：H,W={(h, w)}, k={(kh, kw)}, "
                             f"s={(sh, sw)}, padding={(ph, pw)}")
        xp = F.pad(x, (pw, pw, ph, ph), value=0.0)
        win = xp.unfold(2, kh, sh).unfold(3, kw, sw)  # (N, C, oh, ow, kh, kw)
        return win.abs().pow(pf).sum(dim=(-2, -1)).pow(1.0 / pf)

    if mode == "maxpool_idx":
        if x.dim() != 4:
            raise ValueError(f"maxpool_idx 要求 x 恰 4 维 (N,C,H,W)，got {tuple(x.shape)}")
        if 2 * ph > kh or 2 * pw > kw:
            raise ValueError(f"padding 每边须 <= kernel/2，got padding={(ph, pw)}, "
                             f"kernel={(kh, kw)}")
        n, c, h, w = x.shape
        oh = (h + 2 * ph - kh) // sh + 1
        ow = (w + 2 * pw - kw) // sw + 1
        if oh < 1 or ow < 1:
            raise ValueError(f"窗口放不进输入：H,W={(h, w)}, k={(kh, kw)}, "
                             f"s={(sh, sw)}, padding={(ph, pw)}")
        xp = F.pad(x, (pw, pw, ph, ph), value=float("-inf"))
        win = xp.unfold(2, kh, sh).unfold(3, kw, sw)  # (N, C, oh, ow, kh, kw)
        out = win.amax(dim=(-2, -1))
        # 窗口内行优先展平后取首个最大（与原生并列口径一致）
        local = win.reshape(n, c, oh, ow, kh * kw).argmax(dim=-1)
        # 窗口内局部偏移 → 原输入（未填充）空间图的行优先展平偏移
        rows = (torch.arange(oh) * sh).view(1, 1, oh, 1) + local // kw
        cols = (torch.arange(ow) * sw).view(1, 1, 1, ow) + local % kw
        return out, (rows - ph) * w + (cols - pw)

    # mode == "maxunpool"
    if x.dim() != 4:
        raise ValueError(f"maxunpool 要求 x 恰 4 维 (N,C,h,w)，got {tuple(x.shape)}")
    if sh < kh or sw < kw:
        raise ValueError(f"maxunpool 要求各维 stride >= kernel（重叠写回行为原生未定义），"
                         f"got stride={(sh, sw)}, kernel={(kh, kw)}")
    if ph != 0 or pw != 0:
        raise ValueError(f"maxunpool 要求 padding == 0，got {(ph, pw)}")
    if not torch.is_tensor(indices) or tuple(indices.shape) != tuple(x.shape):
        got = tuple(indices.shape) if torch.is_tensor(indices) else repr(type(indices))
        raise ValueError(f"indices 形状须与 x 一致 {tuple(x.shape)}，got {got}")
    if indices.dtype == torch.bool or indices.dtype.is_floating_point or indices.is_complex():
        raise ValueError(f"indices 必须是整数 dtype，got {indices.dtype}")
    oh, ow = _unpool_out_size(x.shape[-2:], (kh, kw), (sh, sw), output_size)
    n, c, h_s, w_s = x.shape
    flat_idx = indices.to(torch.long).reshape(n, c, h_s * w_s)
    if flat_idx.numel() > 0 and (int(flat_idx.min()) < 0 or int(flat_idx.max()) >= oh * ow):
        raise ValueError(f"indices 越界：取值须落在 [0, {oh * ow})")
    y = torch.zeros(n, c, oh * ow, dtype=x.dtype)
    y.scatter_(2, flat_idx, x.reshape(n, c, h_s * w_s))
    return y.view(n, c, oh, ow)
