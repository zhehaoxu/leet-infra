import torch


def _rank_desc(values):
    """降序、并列取小下标，返回下标列表。"""
    return torch.argsort(-values, stable=True).tolist()


def solve(attn, k, p):
    """A2S 自适应稀疏（float64 累计、int64/bool 输出）。"""
    k = int(k)
    p = float(p)
    if k < 1:
        raise ValueError("k 必须 >= 1")
    if not (0.0 <= p <= 1.0):
        raise ValueError("p 必须在 [0,1] 内")
    steps, length = attn.shape
    prompt = length - steps
    a = attn.to(torch.float64)

    n_adaptive = torch.zeros(steps, dtype=torch.long)
    keep = torch.zeros(steps, length, dtype=torch.bool)
    prev_candidates = None
    for s in range(steps):
        valid = prompt + s
        if valid <= 0:
            n_adaptive[s] = 0
            prev_candidates = []
            continue
        if prev_candidates is None:
            candidates = _rank_desc(a[s, :valid])[:min(k, valid)]
        else:
            order = _rank_desc(a[s, prev_candidates])
            candidates = [prev_candidates[i] for i in order]
        total = float(a[s, :valid].sum())
        cum, n = 0.0, len(candidates)
        for i, idx in enumerate(candidates, start=1):
            cum += float(a[s, idx])
            if cum >= p * total - 1e-12:
                n = i
                break
        chosen = candidates[:n]
        if chosen:
            keep[s, chosen] = True
        n_adaptive[s] = n
        prev_candidates = _rank_desc(a[s, :valid])[:min(k, valid)]
    return n_adaptive, keep
