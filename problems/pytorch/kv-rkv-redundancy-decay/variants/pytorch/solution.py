import math

import torch
import torch.nn.functional as F


def solve(k, q, v, budget=128, window_size=8, kernel_size=7,
          mix_lambda=0.07, threshold=0.5, retain_direction="last"):
    """R-KV 冗余感知打分与预算驱逐（float64 内部，按输入 dtype 返回）。"""
    if k.dim() != 3 or q.dim() != 3 or v.dim() != 3:
        raise ValueError("k/q/v 必须是三维：k,v:(L,H,D)、q:(window_size,H,D)")
    if k.dtype != q.dtype or k.dtype != v.dtype or k.dtype not in (torch.float32, torch.float64):
        raise ValueError("k/q/v 必须同 dtype 且为 float32/float64")
    if v.shape != k.shape or q.shape[1:] != k.shape[1:]:
        raise ValueError("head 数与 head_dim 必须一致，v 必须与 k 同形")
    for name, value in (("budget", budget), ("window_size", window_size),
                        ("kernel_size", kernel_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} 必须是正整数")
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size 必须是奇数")
    if budget <= window_size:
        raise ValueError("budget 必须大于 window_size")
    if q.shape[0] != window_size:
        raise ValueError("q 必须恰好是最后 window_size 个 query")
    if k.shape[0] < budget:
        raise ValueError("cache_len < budget：上游直接返回、不打分")
    if not 0.0 <= mix_lambda <= 1.0:
        raise ValueError("mix_lambda 必须在 [0, 1]")
    if not -1.0 <= threshold <= 1.0:
        raise ValueError("threshold 必须在 [-1, 1]")
    if retain_direction not in ("last", "first"):
        raise ValueError("retain_direction 只能是 last / first")

    out_dtype = k.dtype
    k64 = k.to(torch.float64).transpose(0, 1)          # (H, L, D)
    v64 = v.to(torch.float64).transpose(0, 1)
    q64 = q.to(torch.float64).transpose(0, 1)          # (H, W, D)
    num_heads, kv_len, head_dim = k64.shape
    n = kv_len - window_size

    attn_w = torch.matmul(q64, k64[:, :n].transpose(1, 2)) / math.sqrt(head_dim)
    attn = attn_w.softmax(dim=-1).mean(dim=1)                      # (H, N)
    attn = F.max_pool1d(attn.unsqueeze(0), kernel_size, stride=1,
                        padding=kernel_size // 2).squeeze(0)

    k_norm = k64 / (k64.norm(dim=-1, keepdim=True) + 1e-8)
    sim = torch.matmul(k_norm, k_norm.transpose(1, 2))             # (H, L, L)
    pos = torch.arange(kv_len)
    sim[:, pos, pos] = 0.0
    idx = torch.where(sim > threshold, pos.view(1, 1, -1), torch.zeros(kv_len, dtype=torch.long))
    rep = idx.max(dim=-1).values if retain_direction == "last" else idx.min(dim=-1).values
    sim.scatter_(-1, rep.unsqueeze(-1), 0.0)                       # 代表位的相似度清零
    redundancy = sim.mean(dim=-2).softmax(dim=-1)[:, :n]           # 先在全长上 softmax 再切

    final = mix_lambda * attn - (1.0 - mix_lambda) * redundancy
    pick = torch.argsort(-final, dim=-1, stable=True)[:, : budget - window_size]
    tail = torch.arange(n, kv_len).expand(num_heads, -1)
    kept = torch.cat([pick, tail], dim=-1).to(torch.int64)         # (H, budget)
    gather = kept.unsqueeze(-1).expand(-1, -1, head_dim)
    return {
        "attn": attn.to(out_dtype),
        "redundancy": redundancy.to(out_dtype),
        "final": final.to(out_dtype),
        "kept": kept.contiguous(),
        "kept_keys": k64.gather(1, gather).to(out_dtype),
        "kept_values": v64.gather(1, gather).to(out_dtype),
    }
