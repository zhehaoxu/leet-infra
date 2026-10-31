import torch
import torch.nn.functional as F

TAU, N_SINK, EPS = 5.0, 4, 1e-6


def solve(q, k, v, cos, sin, keep, group_size=1):
    """CapKV：平均未来 RoPE 的 query anchor + 加权 leverage score（float64 内部）。"""
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        raise ValueError("q/k/v 必须是三维")
    if q.dtype != k.dtype or q.dtype != v.dtype or q.dtype not in (torch.float32, torch.float64):
        raise ValueError("q/k/v 必须同 dtype 且为 float32/float64")
    if tuple(k.shape) != tuple(v.shape):
        raise ValueError("k 与 v 形状必须一致")
    num_tokens, num_kv_heads, head_dim = k.shape
    if q.shape[2] != head_dim or head_dim < 2 or head_dim % 2:
        raise ValueError("head_dim 必须一致且为正偶数")
    if q.shape[0] < 1:
        raise ValueError("q 至少要有 1 个 token")
    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size < 1:
        raise ValueError("group_size 必须是正整数")
    if q.shape[1] != num_kv_heads * group_size:
        raise ValueError("q 的头数必须等于 kv 头数 * group_size")
    if num_tokens <= N_SINK:
        raise ValueError(f"cache 长度必须大于 n_sink={N_SINK}")
    if cos.dim() != 2 or cos.shape != sin.shape or cos.shape[1] != head_dim or cos.shape[0] < 1:
        raise ValueError("cos/sin 必须是同形状的 (P, head_dim)，P >= 1")
    if isinstance(keep, bool) or not isinstance(keep, int) or not N_SINK <= keep <= num_tokens:
        raise ValueError(f"keep 必须是 [{N_SINK}, {num_tokens}] 内的整数")

    dtype = q.dtype
    half = head_dim // 2
    eye = torch.eye(head_dim, dtype=torch.float64)
    rot = torch.zeros(head_dim, head_dim, dtype=torch.float64)
    rot[half:, :half] = torch.eye(half, dtype=torch.float64)
    rot[:half, half:] = -torch.eye(half, dtype=torch.float64)

    # 平均未来 RoPE 矩阵；逐 token 旋转再平均 == 先平均再旋转（线性）
    rope = (cos.to(torch.float64).unsqueeze(1) * eye
            + sin.to(torch.float64).unsqueeze(1) * rot).mean(dim=0)
    anchor = q.to(torch.float64).mean(dim=0) @ rope.T                 # (Hq, D)

    keys = k.to(torch.float64).transpose(0, 1)[:, N_SINK:]           # (Hkv, T', D)
    vals = v.to(torch.float64).transpose(0, 1)[:, N_SINK:]
    keys_rep = keys.repeat_interleave(group_size, dim=0)              # (Hq, T', D)
    vals_rep = vals.repeat_interleave(group_size, dim=0)

    stat = (F.normalize(keys_rep, p=2, dim=-1, eps=1e-12)
            * F.normalize(anchor, p=2, dim=-1, eps=1e-12).unsqueeze(1)
            ).sum(dim=-1).clamp(-1.0, 1.0)
    logits = TAU * stat
    weights = torch.exp(logits - logits.amax(dim=-1, keepdim=True))

    scaled = vals_rep * torch.sqrt(weights + EPS).unsqueeze(-1)
    capacity = torch.eye(head_dim, dtype=torch.float64) + scaled.transpose(1, 2) @ scaled
    solved = torch.linalg.solve(capacity, vals_rep.transpose(1, 2))  # A^-1 v^T
    leverage = (vals_rep.transpose(1, 2) * solved).sum(dim=1)        # v^T A^-1 v

    scores = (weights * leverage).reshape(num_kv_heads, group_size, -1).mean(dim=1)
    scores = F.pad(scores, (N_SINK, 0), value=scores.max().item() + 1.0)
    kept = torch.argsort(scores, dim=-1, descending=True, stable=True)[:, :keep]
    return {
        "statistic": stat.to(dtype),
        "weights": weights.to(dtype),
        "leverage": leverage.to(dtype),
        "scores": scores.to(dtype),
        "kept": kept.sort(dim=-1).values.to(torch.int64).contiguous(),
    }
