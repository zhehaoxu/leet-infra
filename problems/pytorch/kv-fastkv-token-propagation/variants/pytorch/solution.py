import torch


def solve(attn_caches, q_len, tsp_idx, hidden_states, position_ids,
          window_size=8, retain_rate=0.1, tsp_rate=0.2):
    """FastKV 跨层 token 选择性传播（arXiv 2502.01068）：TSP 层挑子集，后续层只看它。"""
    if not isinstance(attn_caches, (list, tuple)) or not attn_caches or any(
            not torch.is_tensor(c) or c.dim() != 2 for c in attn_caches):
        raise ValueError("attn_caches 必须是非空 list/tuple 且每项都是二维 tensor")
    dtype = attn_caches[0].dtype
    if dtype not in (torch.float32, torch.float64) or any(c.dtype != dtype for c in attn_caches):
        raise ValueError("attn_caches 必须同 dtype 且为 float32/float64")
    num_layers, num_kv_heads = len(attn_caches), attn_caches[0].shape[1]
    if any(c.shape[1] != num_kv_heads for c in attn_caches):
        raise ValueError("attn_caches 的 kv head 数必须一致")
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (q_len, tsp_idx, window_size)):
        raise ValueError("q_len/tsp_idx/window_size 必须是整数")
    if q_len < 1 or not 0 <= tsp_idx < num_layers or window_size < 1:
        raise ValueError("要求 q_len>=1、tsp_idx 落在 [0, L)、window_size>=1")
    if any(isinstance(r, bool) or not isinstance(r, (int, float)) or not 0 < r <= 1
           for r in (retain_rate, tsp_rate)):
        raise ValueError("retain_rate/tsp_rate 必须落在 (0, 1]")
    if (not torch.is_tensor(hidden_states) or hidden_states.dim() != 2
            or hidden_states.dtype != dtype or hidden_states.shape[0] != q_len):
        raise ValueError("hidden_states 必须是 (q_len, D) 且与 cache 同 dtype")
    if (not torch.is_tensor(position_ids) or position_ids.dtype != torch.int64
            or position_ids.dim() != 1 or position_ids.shape[0] != q_len):
        raise ValueError("position_ids 必须是长度 q_len 的一维 int64 tensor")
    if attn_caches[0].shape[0] != q_len:
        raise ValueError("第 0 层 cache 长度必须等于 q_len")

    tsp_length = int(q_len * tsp_rate)                       # 比例预算向下取整
    gating = q_len > tsp_length                              # 与上游一致：严格大于才传播
    if gating and tsp_length - window_size <= 0:
        raise ValueError("tsp_length - window_size 必须为正")
    capacities, kept, tsp = [], [], None
    for layer, cache in enumerate(attn_caches):
        rate = retain_rate if layer <= tsp_idx else retain_rate / tsp_rate
        length = q_len if (layer <= tsp_idx or not gating) else tsp_length
        if tuple(cache.shape) != (length, num_kv_heads):
            raise ValueError(f"第 {layer} 层 cache 形状应为 {(length, num_kv_heads)}")
        capacity = int(length * rate)
        if capacity - window_size <= 0:
            raise ValueError(f"第 {layer} 层容量 {capacity} 不大于 window_size")
        capacities.append(capacity)
        win = torch.arange(length - window_size, length, dtype=torch.int64)
        if length < capacity:                                # 容量比缓存还大：原样保留
            kept.append(torch.arange(length).expand(num_kv_heads, -1).contiguous())
            continue
        scores = cache.to(torch.float64)
        rank = torch.argsort(scores[:length - window_size], dim=0, descending=True, stable=True)
        rows = torch.cat([rank[:capacity - window_size].T, win.expand(num_kv_heads, -1)], dim=1)
        kept.append(rows.sort(dim=1).values.contiguous())
        if layer == tsp_idx and gating:
            total = scores[:length - window_size].sum(dim=1)  # 跨 kv head 求和
            picked = torch.argsort(total, descending=True, stable=True)[:tsp_length - window_size]
            tsp = torch.cat([picked, win]).sort().values.contiguous()

    hidden_out = None if tsp is None else hidden_states.index_select(0, tsp)
    position_out = None if tsp is None else position_ids.index_select(0, tsp)
    return {
        "tsp_indices": tsp,
        "hidden_out": hidden_out,
        "position_out": position_out,
        "capacities": torch.tensor(capacities, dtype=torch.int64),
        "kept": kept,
    }
