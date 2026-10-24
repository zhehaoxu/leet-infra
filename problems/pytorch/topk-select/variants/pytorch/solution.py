"""参考答案：沿最后一维取 top-k（值 + 索引，降序），用「重复 argmax 后屏蔽」的选择法。

每一轮取当前最大值的位置，记录其值与索引，再把该位置置为 -inf 从候选中移除，
重复 k 次即得到降序的 top-k。返回 (values, indices)，与 torch.topk(x, k, sorted=True) 对齐。
"""

import torch


def solve(x, k):
    xs = x.clone()
    values = x.new_zeros(x.shape[:-1] + (k,))
    indices = torch.zeros(x.shape[:-1] + (k,), dtype=torch.long, device=x.device)
    for j in range(k):
        m = xs.argmax(dim=-1).unsqueeze(-1)          # (..., 1)
        values[..., j] = xs.gather(-1, m).squeeze(-1)
        indices[..., j] = m.squeeze(-1)
        xs.scatter_(-1, m, torch.full_like(m, float("-inf"), dtype=xs.dtype))
    return values, indices
