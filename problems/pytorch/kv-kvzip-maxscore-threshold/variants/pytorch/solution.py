import torch


def _uniform_topk(scores, ratio):
    """每 (layer, head) 各自取 int(N*ratio) 个最高分，同分取更小的列号。"""
    keep = int(scores.shape[-1] * ratio)
    valid = torch.zeros_like(scores, dtype=torch.bool)
    if keep > 0:
        order = torch.argsort(-scores, dim=-1, stable=True)[..., :keep]
        valid.scatter_(-1, order, True)
    return valid


def solve(attn_weights, ratio=0.3, level="pair"):
    """KVzip 跨查询聚合分数 + 全局阈值裁剪（float64 内部，张量按输入 dtype 返回）。"""
    if attn_weights.dim() != 5:
        raise ValueError("attn_weights 必须是五维 (L, H_kv, G, Q, N)")
    if attn_weights.dtype not in (torch.float32, torch.float64):
        raise ValueError("attn_weights 必须是 float32/float64")
    if min(attn_weights.shape) < 1:
        raise ValueError("每个维度都必须 >= 1")
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
        raise ValueError("ratio 必须是实数")
    if ratio < 0:
        raise ValueError("ratio 必须 >= 0")
    if level not in ("pair", "pair-uniform"):
        raise ValueError('level 只能是 "pair" 或 "pair-uniform"')

    out_dtype = attn_weights.dtype
    scores = attn_weights.to(torch.float64).amax(dim=(2, 3))   # 跨 GQA group 与 query 取 max
    total = scores.numel()
    index = 0
    threshold = 0.0

    if level == "pair-uniform":
        valid = _uniform_topk(scores, ratio)
    elif ratio < 1:
        index = max(int(total * ratio) - 1, 0)                 # 下标，不是个数
        threshold = float(torch.sort(scores.reshape(-1), descending=True).values[index])
        valid = scores > threshold                             # 严格大于
    else:
        valid = torch.ones_like(scores, dtype=torch.bool)

    return {
        "scores": scores.to(out_dtype),
        "valid": valid,
        "keep_count": valid.sum(-1).to(torch.int64),
        "threshold": threshold,
        "threshold_index": torch.tensor(index, dtype=torch.int64),
        "keep_ratio": float(valid.to(torch.float64).mean()),
    }
