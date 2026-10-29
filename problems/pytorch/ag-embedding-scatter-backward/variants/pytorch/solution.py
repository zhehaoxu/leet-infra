"""参考答案：一次 index_add_ 完成散点累加，词频摊薄在累加之前做。

gather 的反向天然是 scatter-add，重复下标自动累加，无需先 unique。scale_grad_by_freq
要在**逐 token** 层面除以该 id 的出现次数（bincount 得到），先累加再除会算错权重，
因为同一行的各次贡献次数不同。padding 行直接置零即可：它在前向是常量。
稀疏输出用 torch.sparse_coo_tensor(...).coalesce()，稠密/稀疏两条路必须给同一个值。
"""

import torch


def solve(grad_out, idx, num_embeddings, padding_idx=None, scale_grad_by_freq=False, sparse=False):
    if grad_out.dim() != 2:
        raise ValueError("grad_out 必须是 (n, dim) 二维张量")
    if idx.dim() != 1 or idx.shape[0] != grad_out.shape[0]:
        raise ValueError("idx 必须是与 grad_out 行数等长的一维张量")
    if not isinstance(num_embeddings, int) or num_embeddings <= 0:
        raise ValueError("num_embeddings 必须是正整数")
    if idx.numel():
        lo = int(idx.min())
        hi = int(idx.max())
        if lo < 0 or hi >= num_embeddings:
            raise ValueError("idx 越界：必须落在 [0, num_embeddings)")

    n, d = grad_out.shape
    g = grad_out.double()
    ii = idx.long()

    if scale_grad_by_freq and n:
        cnt = torch.bincount(ii, minlength=num_embeddings).double()
        denom = cnt[ii].clamp(min=1.0)
        g = g / denom.unsqueeze(1)

    w = torch.zeros(num_embeddings, d, dtype=torch.float64)
    if n:
        w.index_add_(0, ii, g)
    if padding_idx is not None:
        w[int(padding_idx)] = 0.0

    if not sparse:
        return w.to(grad_out.dtype)

    rows = (w.abs().sum(dim=1) > 0).nonzero(as_tuple=True)[0]
    cols = torch.arange(d, dtype=torch.long).repeat(rows.numel())
    ri = rows.repeat_interleave(d)
    values = w[rows].reshape(-1).to(grad_out.dtype)
    indices = torch.stack([ri, cols], dim=0).to(torch.long)
    return torch.sparse_coo_tensor(indices, values, size=(num_embeddings, d)).coalesce()
