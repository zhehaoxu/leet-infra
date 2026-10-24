import torch


def solve(scores, chunk_length, compression_ratio):
    """ChunkPress：逐 chunk 段内 top-k，段顺序拼接、段内按分数降序。"""
    H, L = scores.shape
    rows = []
    for h in range(H):
        parts = []
        start = 0
        while start < L:
            seg_len = min(chunk_length, L - start)
            n_kept = max(1, int(seg_len * (1 - compression_ratio)))
            seg = scores[h, start:start + seg_len]
            # 分数降序、平局位置小者优先 → 用 (-score, pos) 稳定排序
            order = sorted(range(seg_len), key=lambda j: (-float(seg[j]), j))
            parts.extend(order[k] + start for k in range(n_kept))
            start += seg_len
        rows.append(parts)
    return torch.tensor(rows, dtype=torch.long)
