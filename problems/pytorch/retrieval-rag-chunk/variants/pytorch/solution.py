"""参考答案：arange 起点 + 列偏移广播，越界处填 -1。"""

import torch


def solve(num_tokens, chunk_size, overlap):
    step = chunk_size - overlap
    starts = torch.arange(0, num_tokens, step)         # (nc,) 每块起点
    cols = torch.arange(chunk_size)                     # (cs,)
    idx = starts[:, None] + cols[None, :]               # (nc, cs)
    return torch.where(idx < num_tokens, idx, torch.full_like(idx, -1)).to(torch.long)
