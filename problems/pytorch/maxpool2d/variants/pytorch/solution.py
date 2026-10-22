"""参考答案：2D 最大池化 —— unfold 出窗口块后取 amax；padding 必须填 -inf。"""

import torch
import torch.nn.functional as F


def _pair(v):
    return (v, v) if isinstance(v, int) else tuple(v)


def solve(x: torch.Tensor, kernel_size, stride=None, padding=0) -> torch.Tensor:
    kh, kw = _pair(kernel_size)
    sh, sw = _pair(stride) if stride is not None else (kh, kw)
    ph, pw = _pair(padding)
    if ph or pw:
        # 陷阱：max 池化的 padding 值必须是 -inf，填 0 会在全负输入时出错
        x = F.pad(x, (pw, pw, ph, ph), value=float("-inf"))
    patches = x.unfold(2, kh, sh).unfold(3, kw, sw)  # (N,C,nh,nw,kh,kw)
    return patches.amax(dim=(4, 5))
