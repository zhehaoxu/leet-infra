"""参考答案：把 NaN 掩成中性元（加法用 0、max 用 -inf、min 用 +inf），再单独数一遍有效元素。

三个高频错点：
1. mean 的分母用 numel() 而不是「非 NaN 个数」——带缺失的数据会被系统性压低，
   做 loss 归约时会把有效 batch 的梯度整体缩小。
2. 直接用 x.sum()：一个 NaN 就把整段泼成 NaN，等于「一处坏样本毒死一个 batch」。
3. 全 NaN 段：sum 的正确答案是 0.0（中性元），mean/max/min 才是 NaN（无定义）。
   这一条必须靠 count==0 的特判来区分，不能靠 where 填值顺手带出来。

inf 不参与「跳过」，所以掩码只针对 isnan；如果顺手把 isinf 也掩掉，含 inf 的段会算错。

长度 0 的归约轴（空张量）也归到「全 NaN 段」那一类：count == 0，所以 sum 给 0.0、
其余给 NaN；amax/amin 要带 initial=，否则 torch 会在空轴上直接报错。
"""

import torch

MODES = ('sum', 'mean', 'max', 'min')
NEUTRAL = {'sum': 0.0, 'max': float('-inf'), 'min': float('inf')}


def _norm_dim(x, dim):
    if dim is None:
        return None
    if isinstance(dim, bool) or not isinstance(dim, int):
        raise ValueError("dim 必须是 int 或 None")
    d = dim + x.dim() if dim < 0 else dim
    if not 0 <= d < x.dim():
        raise ValueError("dim 越界")
    return d


def solve(x, dim=None, mode='mean', keepdim=False):
    if not isinstance(x, torch.Tensor):
        raise ValueError("x 必须是 torch.Tensor")
    if x.dtype not in (torch.float32, torch.float64):
        raise ValueError("x 必须是 float32 或 float64 张量")
    if not isinstance(mode, str) or mode not in MODES:
        raise ValueError(f"未知 mode: {mode}")
    if not isinstance(keepdim, bool):
        raise ValueError("keepdim 必须是 bool")
    d = _norm_dim(x, dim)

    keep_shape = x.shape if d is None else (x.shape[:d] + (1,) + x.shape[d + 1:])
    reduced_len = x.numel() if d is None else x.shape[d]
    if reduced_len == 0:
        # 段长为 0 等价于 count == 0：sum 落在中性元 0.0，其余无定义给 NaN
        base = 0.0 if mode == 'sum' else float('nan')
        out = torch.full(keep_shape, base, dtype=x.dtype, device=x.device)
    else:
        nan_mask = torch.isnan(x)
        neutral = 0.0 if mode in ('sum', 'mean') else NEUTRAL[mode]
        filled = torch.where(nan_mask, torch.full_like(x, neutral), x)
        count = (~nan_mask).sum(dim=d, keepdim=True).to(x.dtype)
        if mode in ('sum', 'mean'):
            out = filled.sum(dim=d, keepdim=True)
            if mode == 'mean':
                out = out / count
        elif mode == 'max':
            out = filled.amax(dim=d, keepdim=True)
        else:
            out = filled.amin(dim=d, keepdim=True)
        if mode != 'sum':
            # 全 NaN 段：mean/max/min 无定义 -> NaN；sum 保留中性元 0.0
            out = torch.where(count == 0, torch.full_like(out, float('nan')), out)

    if d is None:
        return out.reshape(())
    return out if keepdim else out.reshape(x.shape[:d] + x.shape[d + 1:])
