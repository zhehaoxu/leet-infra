"""参考答案：全程在 32 位模式空间里做，连「求 ulp 分数」都只是一次取低位。

因为 bf16 网格 = float32 模式的高 16 位，相邻格点在模式空间里恒差 1<<16，所以
r = (u32 & 0xFFFF) / 65536 直接给出残差比例，不需要 frexp、不需要判断倍频边界；
取 hi 也只是「高 16 位加一」。符号位藏在最高位，加一只会增大幅值，正负数共用同一条代码。

NaN 也顺着同一条路径走：指数全 1 的模式加一还是指数全 1，仍是 NaN，所以不必特判；
真正必须不看 u 的是 r == 0 的格点（含 ±0 和 ±inf），此时三种 variant 的 p 都是 0，
`u < p` 恒假，原值自然被保留——这是这套位表示最舒服的地方。

fwd/bwd 的偏置方向靠 p 的分段函数实现：fwd 把 r <= 0.5 的概率压成 0（一律朝零），
bwd 把 r >= 0.5 的概率抬成 1（一律远离零），中间的斜率都是 2，所以两者的期望正好关于 x 对称。
"""

import torch

VARIANTS = ('sr', 'fwd', 'bwd')


def solve(x, u, variant='sr'):
    for name, t in (('x', x), ('u', u)):
        if not isinstance(t, torch.Tensor):
            raise ValueError(f"{name} 必须是 torch.Tensor")
        if t.dtype != torch.float32:
            raise ValueError(f"{name} 必须是 float32 张量")
    if tuple(x.shape) != tuple(u.shape):
        raise ValueError("x 与 u 形状必须一致")
    if not isinstance(variant, str) or variant not in VARIANTS:
        raise ValueError(f"未知 variant: {variant}")
    if u.numel() and (bool((u < 0).any()) or bool((u >= 1).any()) or bool(torch.isnan(u).any())):
        raise ValueError("u 的元素必须落在 [0, 1) 内")

    pat = x.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    r = (pat & 0xFFFF).to(torch.float64) / 65536.0
    noise = u.contiguous().to(torch.float64)
    if variant == 'sr':
        p = r
    elif variant == 'fwd':
        p = torch.where(r <= 0.5, torch.zeros_like(r), 2.0 * r - 1.0)
    else:
        p = torch.where(r >= 0.5, torch.ones_like(r), 2.0 * r)

    hi = (pat >> 16) + 1
    out = torch.where(noise < p, hi, pat >> 16) << 16
    return out.to(torch.int32).view(torch.float32).reshape(tuple(x.shape))
