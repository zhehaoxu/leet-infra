import torch


def _topk_ids(values, k, forbidden):
    """按分数降序、并列取小下标，跳过 forbidden 集合，返回前 k 个下标。"""
    order = torch.argsort(-values, stable=True).tolist()
    out = []
    if k <= 0:
        return out
    for idx in order:
        if idx in forbidden:
            continue
        out.append(idx)
        if len(out) >= k:
            break
    return out


def solve(scores, n_sink, window, k_spec, k_retr, recency_ratio, concentration):
    """FastGen 头型画像 + 差异化预算（float64 统计、bool/int64 输出）。"""
    n_sink, window = int(n_sink), int(window)
    k_spec, k_retr = int(k_spec), int(k_retr)
    heads, length = scores.shape
    if length < 1:
        raise ValueError("L 必须 >= 1")
    if not (0 <= n_sink <= length):
        raise ValueError("n_sink 必须在 [0,L] 内")
    if not (1 <= window <= length):
        raise ValueError("window 必须在 [1,L] 内")
    if k_spec < 0 or k_retr < 0:
        raise ValueError("k 必须 >= 0")
    for name, val in (("recency_ratio", recency_ratio), ("concentration", concentration)):
        if not (0.0 <= float(val) <= 1.0):
            raise ValueError(f"{name} 必须在 [0,1] 内")
    if bool((scores < 0).any()):
        raise ValueError("scores 必须非负")

    s64 = scores.to(torch.float64)
    total = s64.sum(dim=1)
    safe = torch.where(total > 0, total, torch.ones_like(total))
    recency = torch.where(total > 0, s64[:, length - window:].sum(dim=1) / safe,
                          torch.zeros_like(total))
    conc = torch.where(total > 0, s64.max(dim=1).values / safe, torch.zeros_like(total))

    head_type = torch.full((heads,), 2, dtype=torch.long)
    head_type = torch.where(conc >= float(concentration), torch.full_like(head_type, 1), head_type)
    head_type = torch.where(recency >= float(recency_ratio), torch.zeros_like(head_type), head_type)

    keep = torch.zeros(heads, length, dtype=torch.bool)
    sink = set(range(min(n_sink, length)))
    for h in range(heads):
        t = int(head_type[h])
        if t == 0:
            keep[h, :min(n_sink, length)] = True
            keep[h, max(length - window, 0):] = True
            continue
        k = k_spec if t == 1 else k_retr
        keep[h, :min(n_sink, length)] = True
        chosen = _topk_ids(s64[h], k, sink)
        if chosen:
            keep[h, chosen] = True
    return head_type, keep, keep.sum(dim=1)
