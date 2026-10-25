"""参考答案：手工构造 CSR 三数组（indptr / indices / values，行主序）。"""

import torch


def solve(dense):
    nz = dense != 0.0
    rows, cols = torch.nonzero(nz, as_tuple=True)  # 行主序（行列均升序）
    counts = nz.sum(dim=1).to(torch.long)
    indptr = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
    indices = cols.to(torch.long)
    values = dense[rows, cols].to(torch.float32)
    return indptr, indices, values
