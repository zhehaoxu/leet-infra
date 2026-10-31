import torch


def solve(q, k, page_size=16, page_budget=1):
    """Quest 上界打分 + 末页恒载的预算选页（float64 内部，按输入 dtype 返回）。"""
    if q.dim() != 3 or k.dim() != 3:
        raise ValueError("q/k 必须是三维 (L, H, D)")
    if q.shape[0] != 1:
        raise ValueError("只支持 decode 单 query：Lq 必须等于 1")
    if k.shape[0] < 1:
        raise ValueError("kv 长度必须 >= 1")
    if q.shape[1] != k.shape[1] or q.shape[2] != k.shape[2]:
        raise ValueError("head 数与 head_dim 必须一致")
    if q.dtype != k.dtype or q.dtype not in (torch.float32, torch.float64):
        raise ValueError("q/k 必须同 dtype 且为 float32/float64")
    for name, value in (("page_size", page_size), ("page_budget", page_budget)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} 必须是 int")
        if value < 1:
            raise ValueError(f"{name} 必须 >= 1")

    out_dtype = q.dtype
    kv_len = int(k.shape[0])
    num_pages = -(-kv_len // page_size)
    if num_pages < 2:
        raise ValueError("页数必须 >= 2（最后一页要被排除）")

    q_row = q.to(torch.float64).reshape(-1, q.shape[2])      # (H, D)
    k64 = k.to(torch.float64).transpose(0, 1).contiguous()   # (H, Lkv, D)
    num_heads, head_dim = q_row.shape

    sign = torch.where(q_row > 0, 1.0, -1.0)                 # q == 0 归负
    max_key = k64 * sign.unsqueeze(1)
    pad = num_pages * page_size - kv_len
    if pad:
        sentinel = torch.full((num_heads, pad, head_dim),
                              torch.finfo(out_dtype).min, dtype=torch.float64)
        max_key = torch.cat([max_key, sentinel], dim=1)
    page_max = max_key.reshape(num_heads, num_pages, page_size, head_dim).amax(dim=2)

    scores = torch.matmul((q_row * sign).unsqueeze(1), page_max.transpose(1, 2))
    scores = scores.squeeze(1)[:, : num_pages - 1].contiguous()   # (H, P-1)

    keep = min(page_budget, num_pages) - 1
    last_page = torch.full((num_heads, 1), num_pages - 1, dtype=torch.int64)
    if keep == 0:
        selected = last_page
    else:
        order = torch.argsort(-scores, dim=-1, stable=True)      # 同分保留小页号
        picked = order[:, :keep].to(torch.int64)
        selected = torch.cat([picked, last_page], dim=-1).sort(dim=-1).values

    return {
        "sign": sign.to(out_dtype),
        "page_max": page_max.to(out_dtype),
        "scores": scores.to(out_dtype),
        "selected": selected.contiguous(),
    }
