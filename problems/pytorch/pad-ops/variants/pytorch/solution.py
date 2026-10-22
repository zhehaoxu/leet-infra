"""参考答案：常量 Padding —— padding 从最后一个维向前成对(左,右)生效，中间填入 value。"""

import torch


def solve(x: torch.Tensor, padding: tuple[int, ...], mode: str = "constant", value: float = 0.0) -> torch.Tensor:
    if mode != "constant":
        raise ValueError("本题仅支持 constant 模式")
    lefts = [0] * x.dim()
    rights = [0] * x.dim()
    for i in range(len(padding) // 2):
        d = x.dim() - 1 - i                     # 从最后一维向前
        lefts[d] = padding[2 * i]
        rights[d] = padding[2 * i + 1]
    out_shape = [s + l + r for s, l, r in zip(x.shape, lefts, rights)]
    out = torch.full(out_shape, value, dtype=x.dtype, device=x.device)
    center = tuple(slice(l, l + s) for l, s in zip(lefts, x.shape))
    out[center] = x
    return out
