"""参考答案：沿最后一维排序并返回 argsort 索引（升序），用选择排序实现，不直接调 torch.sort。

每轮在未排序部分取最小值位置，写出该值与其原始索引，再把该位置置为 +inf 移出候选，
重复到结束。返回 (sorted_values, indices)，与 torch.sort(x, dim=-1) 对齐（平局时取靠前索引）。
"""

import torch


def solve(x):
    n = x.shape[-1]
    xs = x.clone()
    values = x.new_zeros(x.shape)
    indices = torch.zeros(x.shape, dtype=torch.long, device=x.device)
    for i in range(n):
        m = xs.argmin(dim=-1).unsqueeze(-1)           # (..., 1)
        values[..., i] = xs.gather(-1, m).squeeze(-1)
        indices[..., i] = m.squeeze(-1)
        xs.scatter_(-1, m, torch.full_like(m, float("inf"), dtype=xs.dtype))
    return values, indices
