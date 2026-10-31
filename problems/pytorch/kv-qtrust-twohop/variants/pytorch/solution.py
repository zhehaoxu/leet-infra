import torch


def solve(q, k, request_mask, window=16, tau=1.0, recent=16, budget=32):
    """Q-Trust 两跳打分 + 尾部恒保的预算选 key（float64 内部，按输入 dtype 返回）。"""
    if q.dim() != 3 or k.dim() != 3:
        raise ValueError("q/k 必须是三维：q (N,H,D)、k (N,KV,D)")
    seq, num_heads, head_dim = q.shape
    if k.shape[0] != seq or k.shape[2] != head_dim:
        raise ValueError("q/k 的序列长度与 head_dim 必须一致")
    if q.dtype != k.dtype or q.dtype not in (torch.float32, torch.float64):
        raise ValueError("q/k 必须同 dtype 且为 float32/float64")
    num_kv = int(k.shape[1])
    if num_kv < 1 or num_heads % num_kv:
        raise ValueError("H 必须是 KV 的正整数倍（GQA 组要整除）")
    if request_mask.dtype != torch.bool or tuple(request_mask.shape) != (seq,):
        raise ValueError("request_mask 必须是 (N,) 的 bool")
    for name, value, low in (("window", window, 1), ("recent", recent, 0), ("budget", budget, 1)):
        if isinstance(value, bool) or not isinstance(value, int) or value < low:
            raise ValueError(f"{name} 必须是 >= {low} 的整数")
    if isinstance(tau, bool) or not isinstance(tau, (int, float)) or tau < 0:
        raise ValueError("tau 必须是非负浮点数")
    if window > seq:
        raise ValueError("window 不能超过 N")
    if recent > budget or budget >= seq:
        raise ValueError("需要 recent <= budget < N（否则没有可竞标的 key）")

    group = num_heads // num_kv
    out_dtype = q.dtype
    q64 = q.to(torch.float64).permute(1, 0, 2)                       # (H, N, D)
    k64 = k.to(torch.float64).repeat_interleave(group, dim=1)        # (N, H, D)
    logits = torch.matmul(q64, k64.permute(1, 2, 0)) * (head_dim ** -0.5)
    row = torch.arange(seq).unsqueeze(1)
    logits = logits.masked_fill(torch.arange(seq).unsqueeze(0) > row, float("-inf"))
    probs = torch.softmax(logits, dim=-1)                            # (H, N, N) 因果注意力
    grouped = probs.reshape(num_kv, group, seq, seq).sum(dim=1)      # (KV, N, N) 组内求和

    first_hop = grouped[:, seq - window:, :].sum(dim=1)              # (KV, N)
    raw = first_hop.clone()
    raw[:, request_mask] = 1.0                                       # 请求段原始权重恒为 1
    weight = raw / raw.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    if tau > 0:
        keep = weight >= tau / seq
        keep[:, request_mask] = True                                 # 请求段永不剔除
        weight = weight * keep
    else:
        keep = torch.ones_like(weight, dtype=torch.bool)
    score = torch.einsum("kq,kqi->ki", weight, grouped)              # (KV, N)

    head, slots = seq - recent, budget - recent
    tail = torch.arange(head, seq).expand(num_kv, recent)
    if slots:
        picked = torch.argsort(-score[:, :head], dim=-1, stable=True)[:, :slots]
        selected = torch.cat([picked, tail], dim=-1).sort(dim=-1).values
    else:
        selected = tail

    return {
        "first_hop": first_hop.to(out_dtype),
        "query_weight": weight.to(out_dtype),
        "kept_query": keep.any(dim=0).nonzero().flatten().to(torch.int64),
        "score": score.to(out_dtype),
        "selected": selected.to(torch.int64).contiguous(),
    }
