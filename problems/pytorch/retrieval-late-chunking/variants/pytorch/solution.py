"""参考答案：late chunking 池化与父子块扩展。

token_emb 已是上下文化 token 向量：每个块在 attention mask 内做均值池化
（全 mask 掉的块记 0），可选逐行 L2 归一化；父块按 child // parent_size
分组，区间取组内首块起点到末块终点，向量用同一 mask 均值口径在整段上
重算（不是子块向量的平均，避免块长加权偏差）。
"""

import torch


def _l2_normalize_rows(x):
    if x.shape[0] == 0:
        return x
    norm = x.norm(dim=1, keepdim=True)
    safe = torch.where(norm > 0, norm, torch.ones_like(norm))
    return x / safe


def _masked_mean(token_emb, mask, spans):
    """spans 为 [(start, end), ...]，逐段在 mask 内做均值；无有效 token 记 0。"""
    out = token_emb.new_zeros((len(spans), token_emb.shape[1]))
    for k, (s, e) in enumerate(spans):
        if e <= s:
            continue
        m = mask[s:e].to(token_emb.dtype)
        cnt = m.sum()
        if cnt > 0:
            out[k] = (token_emb[s:e] * m.unsqueeze(1)).sum(dim=0) / cnt
    return out


def solve(token_emb, attention_mask, chunk_bounds, hit_chunks, parent_size,
          normalize=True):
    if token_emb.dim() != 2:
        raise ValueError(
            f"token_emb 必须是二维 (L, D)，got {tuple(token_emb.shape)}")
    if not token_emb.is_floating_point():
        raise ValueError(
            f"token_emb 必须是浮点张量，got {token_emb.dtype}")
    if chunk_bounds.dim() != 2 or chunk_bounds.shape[1] != 2:
        raise ValueError(
            f"chunk_bounds 必须是 (C, 2)，got {tuple(chunk_bounds.shape)}")
    if chunk_bounds.is_floating_point() or chunk_bounds.is_complex():
        raise ValueError(
            f"chunk_bounds 必须是整型，got {chunk_bounds.dtype}")
    if hit_chunks.dim() != 1:
        raise ValueError(
            f"hit_chunks 必须是一维，got {tuple(hit_chunks.shape)}")
    if hit_chunks.is_floating_point() or hit_chunks.is_complex():
        raise ValueError(
            f"hit_chunks 必须是整型，got {hit_chunks.dtype}")
    if parent_size < 1:
        raise ValueError(f"parent_size 必须 >= 1，got {parent_size}")

    length = token_emb.shape[0]
    if attention_mask.dim() != 1 or attention_mask.shape[0] != length:
        raise ValueError(
            f"attention_mask 必须是长度 {length} 的一维张量，"
            f"got {tuple(attention_mask.shape)}")

    bounds = chunk_bounds.tolist()
    n_chunks = len(bounds)
    prev_start = -1
    for s, e in bounds:
        if s < 0 or e < 0 or s > e or e > length:
            raise ValueError(f"非法块区间 [{s}, {e})，L={length}")
        if s < prev_start:
            raise ValueError(f"chunk_bounds 必须按 start 非降序，got {bounds}")
        prev_start = s

    hits = [int(v) for v in hit_chunks.tolist()]
    for h in hits:
        if h < 0 or h >= n_chunks:
            raise ValueError(f"命中子块下标 {h} 越界，C={n_chunks}")

    mask = (attention_mask != 0)
    chunk_spans = [(int(s), int(e)) for s, e in bounds]
    chunk_vecs = _masked_mean(token_emb, mask, chunk_spans)
    if normalize:
        chunk_vecs = _l2_normalize_rows(chunk_vecs)

    # 父块：child // parent_size 分组，按父块下标升序去重输出
    parent_ids = sorted({h // parent_size for h in hits})
    parent_spans = []
    for p in parent_ids:
        first = p * parent_size
        last = min((p + 1) * parent_size, n_chunks) - 1
        parent_spans.append((chunk_spans[first][0], chunk_spans[last][1]))

    parent_vecs = _masked_mean(token_emb, mask, parent_spans)
    if normalize:
        parent_vecs = _l2_normalize_rows(parent_vecs)

    parent_bounds = torch.tensor(
        parent_spans, dtype=torch.int64).reshape(len(parent_spans), 2)
    row_of = {p: i for i, p in enumerate(parent_ids)}
    hit_to_parent = torch.tensor(
        [row_of[h // parent_size] for h in hits], dtype=torch.int64)

    return chunk_vecs, parent_bounds, parent_vecs, hit_to_parent
