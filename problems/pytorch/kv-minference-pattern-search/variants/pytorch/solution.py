import math
import torch

LAST_Q = 64          # vertical 统计只看最后 64 个 query
FORCE = 30           # vertical 前 30 列、slash 最后 30 条对角线强制保留
BLOCK = 32           # block_sparse 的块宽
KINDS = ("stream_llm", "vertical_and_slash", "block_sparse")


def _causal(n):
    idx = torch.arange(n)
    return idx[None, :] <= idx[:, None]


def _stream_mask(n, vertical_size, slash_size):
    idx = torch.arange(n)
    band = (idx[None, :] <= idx[:, None]) & (idx[None, :] >= idx[:, None] - slash_size)
    return band | ((idx[None, :] < vertical_size) & (idx[None, :] <= idx[:, None]))


def _diag_sums(tail):
    """tail:(H,64,N) 最后 64 个 query 的注意力 → (H,N)，第 j 项是斜率 delta=j+1-N 那条对角线之和。"""
    h, rows, n = tail.shape
    out = torch.zeros(h, n, dtype=tail.dtype)
    for j in range(n):
        delta = j + 1 - n
        lo = max(0, -delta)
        if lo < rows:
            idx = torch.arange(lo, rows)
            out[:, j] = tail[:, idx, idx + delta].sum(1)
    return out


def _vertical_and_slash(attn, vertical_size, slash_size):
    h, n, _ = attn.shape
    tail = attn[:, n - LAST_Q:, :]
    vertical = tail.sum(-2)
    vertical[:, :FORCE] = torch.inf
    keep = torch.ones(h, n, dtype=torch.bool)
    keep.scatter_(1, torch.argsort(vertical, dim=-1, stable=True)[:, : n - vertical_size], False)

    slash = _diag_sums(tail)
    slash[:, n - FORCE:] = torch.inf
    offsets = torch.argsort(-slash, dim=-1, stable=True)[:, :slash_size] - (n - 1)
    mask = _causal(n) & keep.unsqueeze(1)
    for head in range(h):
        for off in offsets[head].tolist():               # off<0：query 往回看 off 个 key
            row = torch.arange(max(0, -off), min(n, n - off))
            mask[head, row, row + off] = True
    return mask


def _block_sparse(q, k, topk_ratio):
    h, n, dim = q.shape
    nb = (n - 1) // BLOCK + 1
    bq = torch.zeros(h, nb * BLOCK, dim, dtype=q.dtype)
    bk = torch.zeros(h, nb * BLOCK, dim, dtype=k.dtype)
    bq[:, :n], bk[:, :n] = q, k
    bq = bq.reshape(h, nb, BLOCK, dim).mean(2)          # 补零也进分母
    bk = bk.reshape(h, nb, BLOCK, dim).mean(2)
    qk = torch.matmul(bq, bk.transpose(1, 2))           # 上游这里没有 /sqrt(d)
    idx = torch.arange(nb)
    qk = qk.masked_fill(idx[None, :] > idx[:, None], float("-inf"))
    order = torch.argsort(qk, dim=-1, stable=True)[..., : nb - nb // topk_ratio]
    est = torch.ones(h, nb, nb, dtype=torch.bool)
    est.scatter_(2, order, False)                       # 丢掉注意力最低的块
    est = est[:, :, None, :, None].expand(h, nb, BLOCK, nb, BLOCK).reshape(h, nb * BLOCK, nb * BLOCK)
    return (est & _causal(nb * BLOCK))[:, :n, :n]


def solve(q, k, candidates):
    """MInference 稀疏 pattern 搜索（float64 内部，分数按输入 dtype 返回）。"""
    if q.dim() != 3 or k.dim() != 3 or q.shape != k.shape:
        raise ValueError("q/k 必须是同形状三维 (H, N, D)")
    if q.dtype != k.dtype or q.dtype not in (torch.float32, torch.float64):
        raise ValueError("q/k 必须同 dtype 且为 float32/float64")
    if q.shape[1] < LAST_Q or q.shape[2] < 1:
        raise ValueError(f"N 必须 >= {LAST_Q} 且 D >= 1")
    if not isinstance(candidates, (list, tuple)) or not candidates:
        raise ValueError("candidates 不能为空")
    checked = []
    for cand in candidates:
        if not isinstance(cand, (list, tuple)) or len(cand) != 3:
            raise ValueError("candidate 必须是 (kind, a, b)")
        kind, a, b = cand
        if kind not in KINDS:
            raise ValueError(f"未知 kind：{kind}")
        if isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, int) or not isinstance(b, int):
            raise ValueError("candidate 的 a/b 必须是 int")
        if kind == "block_sparse" and a < 1:
            raise ValueError("block_sparse 的 topk_ratio 必须 >= 1")
        if kind != "block_sparse" and not (0 <= a <= q.shape[1] and 0 <= b <= q.shape[1]):
            raise ValueError("vertical/slash 尺寸必须落在 [0, N]")
        checked.append((kind, a, b))

    num_heads, n, dim = q.shape
    q64, k64 = q.to(torch.float64), k.to(torch.float64)
    logits = torch.matmul(q64, k64.transpose(1, 2)) / math.sqrt(dim)
    attn = torch.softmax(logits.masked_fill(~_causal(n), float("-inf")), dim=-1)

    scores, masks = [], []
    for kind, a, b in checked:
        if kind == "stream_llm":
            mask = _stream_mask(n, a, b).expand(num_heads, n, n)
        elif kind == "vertical_and_slash":
            mask = _vertical_and_slash(attn, a, b)
        else:
            mask = _block_sparse(q64, k64, a)
        masks.append(mask)
        scores.append((attn * mask).sum(-1).mean(-1))    # 每行保留的注意力质量
    scores = torch.stack(scores, dim=1)                 # (H, C)
    best = scores.argmax(dim=1)                         # 同分取更小的候选下标

    pattern, winner = torch.zeros(num_heads, 2, dtype=torch.int64), []
    for head in range(num_heads):
        kind, a, b = checked[int(best[head])]
        pattern[head] = torch.tensor((1000, 6096) if kind == "block_sparse" else (a, b))
        winner.append(masks[int(best[head])][head])
    return {"scores": scores.to(q.dtype), "best": best,
            "pattern": pattern, "mask": torch.stack(winner)}
