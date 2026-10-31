import torch


def solve(keys, values, scores, counts, dense, new_key, new_value, new_score, metric="l2"):
    """ZeroMerge 加权合并：挤掉非 dense 最低分槽位，与 dense 区最近邻按计数加权合并。"""
    if keys.dim() != 3 or values.dim() != 3 or new_key.dim() != 2 or new_value.dim() != 2:
        raise ValueError("keys/values 需 (T, H, D)，new_key/new_value 需 (H, D)")
    if keys.shape != values.shape:
        raise ValueError("keys/values 形状必须一致")
    t_slots, n_heads, head_dim = keys.shape
    if scores.shape != (t_slots, n_heads) or counts.shape != (t_slots, n_heads):
        raise ValueError("scores/counts 必须是 (T, H)")
    if dense.shape != (t_slots,) or dense.dtype != torch.bool:
        raise ValueError("dense 必须是 (T,) 的 bool 掩码")
    if (new_key.shape != (n_heads, head_dim) or new_value.shape != (n_heads, head_dim)
            or new_score.shape != (n_heads,)):
        raise ValueError("new_key/new_value 必须是 (H, D)，new_score 必须是 (H,)")
    dtypes = {keys.dtype, values.dtype, scores.dtype, counts.dtype,
              new_key.dtype, new_value.dtype, new_score.dtype}
    if len(dtypes) != 1 or keys.dtype not in (torch.float32, torch.float64):
        raise ValueError("所有浮点输入必须同 dtype 且为 float32/float64")
    if metric not in ("l2", "dot_product"):
        raise ValueError(f"未知 metric：{metric}")
    if bool(dense.all()):
        raise ValueError("dense 区不能覆盖全部槽位：没有可挤掉的槽位")
    if bool((counts <= 0).any()):
        raise ValueError("counts 必须全为正")

    out_dtype = keys.dtype
    k64, v64 = keys.to(torch.float64), values.to(torch.float64)
    s64, c64 = scores.to(torch.float64), counts.to(torch.float64)
    nk, nv = new_key.to(torch.float64), new_value.to(torch.float64)
    ns = new_score.to(torch.float64)
    rows = torch.arange(n_heads)
    dense_idx = dense.nonzero(as_tuple=True)[0]
    out_k, out_v = k64.clone(), v64.clone()
    out_c, out_s = c64.clone(), s64.clone()

    if dense_idx.numel() == 0:
        # 上游 cache_dense == 0 分支：不合并，全局最低分槽位直接让给新 token。
        pick = torch.argsort(s64, dim=0, stable=True)[0]
        j_slot = torch.full_like(pick, -1)
    else:
        dyn_idx = (~dense).nonzero(as_tuple=True)[0]
        pick = dyn_idx[torch.argsort(s64[dyn_idx], dim=0, stable=True)[0]]
        k_i = k64[pick, rows]
        cand = k64[dense_idx]
        if metric == "l2":
            dist = ((cand - k_i.unsqueeze(0)) ** 2).sum(dim=-1)
        else:
            dist = -(cand * k_i.unsqueeze(0)).sum(dim=-1)
        j_slot = dense_idx[torch.argsort(dist, dim=0, stable=True)[0]]
        w_i = c64[pick, rows].unsqueeze(-1)
        w_j = c64[j_slot, rows].unsqueeze(-1)
        total = w_i + w_j
        out_k[j_slot, rows] = (w_i * k_i + w_j * k64[j_slot, rows]) / total
        out_v[j_slot, rows] = (w_i * v64[pick, rows] + w_j * v64[j_slot, rows]) / total
        out_c[j_slot, rows] = c64[pick, rows] + c64[j_slot, rows]
        out_s[j_slot, rows] = s64[pick, rows] + s64[j_slot, rows]

    out_k[pick, rows], out_v[pick, rows] = nk, nv
    out_c[pick, rows], out_s[pick, rows] = 1.0, ns

    return {
        "keys": out_k.to(out_dtype),
        "values": out_v.to(out_dtype),
        "counts": out_c.to(out_dtype),
        "scores": out_s.to(out_dtype),
        "merged_slots": torch.stack([pick, j_slot], dim=-1).to(torch.int64),
    }
