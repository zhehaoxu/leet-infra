import torch


def solve(scores, chunk_length, compression_ratio):
    """ChunkKV：块级分数 = 跨 head 求和后块内平均；按块分 top-k 整块保留、升序合并。"""
    H, L = scores.shape
    c = int(chunk_length)
    tok_sum = scores.sum(dim=0)  # (L,) 跨 head 求和
    n_chunks = (L + c - 1) // c
    cs = []
    for i in range(n_chunks):
        lo = i * c
        hi = min(lo + c, L)
        cs.append(float(tok_sum[lo:hi].mean()))
    n_kept_chunks = max(1, int(n_chunks * (1 - compression_ratio)))
    order = sorted(range(n_chunks), key=lambda i: (-cs[i], i))
    kept_chunks = order[:n_kept_chunks]
    idxs = []
    for i in kept_chunks:
        lo = i * c
        hi = min(lo + c, L)
        idxs.extend(range(lo, hi))
    return torch.tensor(sorted(idxs), dtype=torch.long)
