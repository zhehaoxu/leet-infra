"""参考答案：ColBERT MaxSim 与 MaxPooled 长文档聚合。

先对 query / doc 逐行做 L2 归一化（零行保持 0），得到余弦矩阵
cos[i, j]，再沿 query 词求和 max_j cos。chunk 按 chunk_size 顺序硬切、
尾块保留；逐 chunk 取 max 得到 (Lq, C) 矩阵。maxpooled 对 chunk 维再取
max（每个 query 词只计一次），repeated 对 chunk 维求和（重复计分口径）。
"""

import torch


def _l2_normalize_rows(x):
    """逐行 L2 归一化；范数为 0 的行保持全 0（不产生 NaN）。"""
    norm = x.norm(dim=1, keepdim=True)
    safe = torch.where(norm > 0, norm, torch.ones_like(norm))
    return x / safe


def solve(query, doc, chunk_size):
    if query.dim() != 2:
        raise ValueError(f"query 必须是二维 (Lq, D)，got {tuple(query.shape)}")
    if doc.dim() != 2:
        raise ValueError(f"doc 必须是二维 (Ld, D)，got {tuple(doc.shape)}")
    if query.dtype != doc.dtype:
        raise ValueError(
            f"query 与 doc 必须同 dtype，got {query.dtype} vs {doc.dtype}")
    if not query.is_floating_point():
        raise ValueError(f"query / doc 必须是浮点张量，got {query.dtype}")
    if query.shape[1] != doc.shape[1]:
        raise ValueError(
            f"特征维必须一致，got {query.shape[1]} vs {doc.shape[1]}")
    if chunk_size < 1:
        raise ValueError(f"chunk_size 必须 >= 1，got {chunk_size}")

    lq, ld = query.shape[0], doc.shape[0]
    qn = _l2_normalize_rows(query)
    dn = _l2_normalize_rows(doc)

    if ld == 0:
        cos = query.new_zeros((lq, 0))
        base = query.new_zeros(())
        base_idx = torch.full((lq,), -1, dtype=torch.int64)
    else:
        cos = qn @ dn.transpose(0, 1)                 # (Lq, Ld)
        max_val = cos.max(dim=1, keepdim=True).values
        base = max_val.squeeze(1).sum()
        # 并列时取最小下标：布尔矩阵的第一个 True 即首次出现
        base_idx = (cos == max_val).to(torch.int64).argmax(dim=1)

    n_chunks = (ld + chunk_size - 1) // chunk_size
    chunk_maxsim = query.new_zeros((lq, n_chunks))
    for c in range(n_chunks):
        lo = c * chunk_size
        hi = min(lo + chunk_size, ld)
        chunk_maxsim[:, c] = cos[:, lo:hi].max(dim=1).values

    if n_chunks == 0:
        maxpooled = query.new_zeros(())
        repeated = query.new_zeros(())
    else:
        maxpooled = chunk_maxsim.max(dim=1).values.sum()
        repeated = chunk_maxsim.sum()

    return base, base_idx, chunk_maxsim, maxpooled, repeated
