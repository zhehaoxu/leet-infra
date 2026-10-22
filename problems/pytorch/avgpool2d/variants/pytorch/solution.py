"""参考答案：2D 平均池化 —— unfold 求和；count_include_pad=False 时除以有效像素数。"""

import torch
import torch.nn.functional as F


def _pair(v):
    return (v, v) if isinstance(v, int) else tuple(v)


def solve(x: torch.Tensor, kernel_size, stride=None, padding=0, count_include_pad: bool = True) -> torch.Tensor:
    kh, kw = _pair(kernel_size)
    sh, sw = _pair(stride) if stride is not None else (kh, kw)
    ph, pw = _pair(padding)
    ones = torch.ones_like(x)
    if ph or pw:
        x = F.pad(x, (pw, pw, ph, ph), value=0.0)
        ones = F.pad(ones, (pw, pw, ph, ph), value=0.0)
    s = x.unfold(2, kh, sh).unfold(3, kw, sw).sum(dim=(4, 5))
    if count_include_pad:
        return s / (kh * kw)
    # 有效像素数：对同样 0-pad 的补 1 掩码再走一遍窗口求和
    c = ones.unfold(2, kh, sh).unfold(3, kw, sw).sum(dim=(4, 5))
    return s / c
