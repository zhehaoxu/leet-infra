import torch


def solve(attn, keys, values, max_capacity, window_size, imp_budget=0.5,
          fuse_temperature=1.0, value_norms=None):
    """MorphKV 层级岛屿融合：重要 token 留单例，其余连续岛按 importance 加权融成 1 槽。"""
    if attn.dim() != 3 or keys.dim() != 3 or values.dim() != 3:
        raise ValueError("attn 需 (H, Q, P)，keys/values 需 (H, P+W, D)")
    if keys.shape != values.shape:
        raise ValueError("keys/values 形状必须一致")
    if not isinstance(max_capacity, int) or not isinstance(window_size, int):
        raise ValueError("max_capacity/window_size 必须是 int")
    n_heads, n_query, n_past = attn.shape
    head_dim = keys.shape[2]
    if keys.shape[0] != n_heads:
        raise ValueError("attn 与 keys/values 的 head 数必须一致")
    if n_past < 1 or n_query < 1 or head_dim < 1:
        raise ValueError("P / Q / D 必须为正")
    if window_size < 0 or max_capacity < 1:
        raise ValueError("window_size 需 >= 0，max_capacity 需 >= 1")
    if keys.shape[1] != n_past + window_size:
        raise ValueError("keys/values 的第二维必须等于 P + window_size")
    if not 0.0 <= imp_budget <= 1.0:
        raise ValueError("imp_budget 必须落在 [0, 1]")
    if fuse_temperature <= 0:
        raise ValueError("fuse_temperature 必须为正")
    dtypes = {attn.dtype, keys.dtype, values.dtype}
    if value_norms is not None:
        dtypes.add(value_norms.dtype)
    if len(dtypes) != 1 or keys.dtype not in (torch.float32, torch.float64):
        raise ValueError("所有浮点输入必须同 dtype 且为 float32/float64")
    if value_norms is not None and value_norms.shape != (n_heads, n_past):
        raise ValueError("value_norms 必须是 (H, P)")

    # 预算：上游 num_important 先取整、再夹到 num_past；num_fused_slots 用未夹的预算算。
    n_important_budget = int(max_capacity * imp_budget)
    n_fused_slots = max_capacity - window_size - n_important_budget
    if n_fused_slots < 0:
        raise ValueError("预算不足：max_capacity 必须 >= window_size + int(max_capacity*imp_budget)")
    n_important = min(n_important_budget, n_past)
    n_unimportant = n_past - n_important
    n_islands = n_important + 1

    out_dtype = keys.dtype
    logits = attn.to(torch.float64) / float(fuse_temperature)
    if value_norms is None:
        norms = values[:, :n_past, :].to(torch.float64).norm(dim=-1)
    else:
        norms = value_norms.to(torch.float64)
    if bool((norms <= 0).any()):
        raise ValueError("value_norms 必须全为正")
    importance = torch.softmax(logits, dim=-1).mean(dim=-2) * norms  # (H, P)

    # 重要 token：按重要性降序取前 n_important，同分取更小的位置号。
    order = torch.argsort(importance, dim=-1, descending=True, stable=True)
    important_idx = order[:, :n_important].sort(dim=-1).values
    is_important = torch.zeros(n_heads, n_past, dtype=torch.bool)
    is_important.scatter_(1, important_idx, True)
    unimportant_idx = (
        torch.arange(n_past).expand(n_heads, n_past)[~is_important].view(n_heads, n_unimportant)
    )

    # 岛屿号 = 该位置之前（含自身，但自身非重要）的重要 token 个数。
    island_ids_all = torch.cumsum(is_important.to(torch.int64), dim=-1)
    unimp_islands = island_ids_all.gather(1, unimportant_idx)
    unimp_importance = importance.gather(1, unimportant_idx)

    past_keys = keys[:, :n_past, :].to(torch.float64)
    past_vals = values[:, :n_past, :].to(torch.float64)
    exp_idx = unimportant_idx.unsqueeze(-1).expand(-1, -1, head_dim)
    unimp_keys = past_keys.gather(1, exp_idx)
    unimp_vals = past_vals.gather(1, exp_idx)

    island_mass = torch.zeros(n_heads, n_islands, dtype=torch.float64)
    island_mass.scatter_add_(1, unimp_islands, unimp_importance)
    island_counts = torch.zeros(n_heads, n_islands, dtype=torch.int64)
    island_counts.scatter_add_(1, unimp_islands, torch.ones_like(unimp_islands))

    weights = unimp_importance / island_mass.gather(1, unimp_islands).clamp(min=1e-8)
    fused_keys = torch.zeros(n_heads, n_islands, head_dim, dtype=torch.float64)
    fused_vals = torch.zeros(n_heads, n_islands, head_dim, dtype=torch.float64)
    island_exp = unimp_islands.unsqueeze(-1).expand(-1, -1, head_dim)
    fused_keys.scatter_add_(1, island_exp, unimp_keys * weights.unsqueeze(-1))
    fused_vals.scatter_add_(1, island_exp, unimp_vals * weights.unsqueeze(-1))

    # 岛屿打分 = 岛内平均重要性；空岛记 -inf，按分数降序（同分取更小的岛号）取 top-K。
    island_scores = island_mass / island_counts.clamp(min=1).to(torch.float64)
    island_scores = island_scores.masked_fill(island_counts == 0, float("-inf"))
    n_sel = min(n_fused_slots, n_islands)
    selected = torch.argsort(island_scores, dim=-1, descending=True, stable=True)[:, :n_sel]
    selected = selected.sort(dim=-1).values

    sel_exp = selected.unsqueeze(-1).expand(-1, -1, head_dim)
    sel_keys = fused_keys.gather(1, sel_exp)
    sel_vals = fused_vals.gather(1, sel_exp)
    sel_counts = island_counts.gather(1, selected)

    n_pad = n_fused_slots - n_sel
    if n_pad > 0:
        sel_keys = torch.cat([sel_keys, torch.zeros(n_heads, n_pad, head_dim, dtype=torch.float64)], dim=1)
        sel_vals = torch.cat([sel_vals, torch.zeros(n_heads, n_pad, head_dim, dtype=torch.float64)], dim=1)
        sel_counts = torch.cat([sel_counts, torch.zeros(n_heads, n_pad, dtype=torch.int64)], dim=1)
        selected = torch.cat([selected, torch.full((n_heads, n_pad), -1, dtype=torch.int64)], dim=1)

    imp_exp = important_idx.unsqueeze(-1).expand(-1, -1, head_dim)
    imp_keys = past_keys.gather(1, imp_exp)
    imp_vals = past_vals.gather(1, imp_exp)
    window_keys = keys[:, n_past:, :].to(torch.float64)
    window_vals = values[:, n_past:, :].to(torch.float64)

    return {
        "keys": torch.cat([sel_keys, imp_keys, window_keys], dim=1).to(out_dtype),
        "values": torch.cat([sel_vals, imp_vals, window_vals], dim=1).to(out_dtype),
        "important_idx": important_idx,
        "island_idx": selected,
        "island_counts": sel_counts,
    }
