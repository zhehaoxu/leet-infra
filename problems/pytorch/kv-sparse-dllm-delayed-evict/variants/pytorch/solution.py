import torch
import torch.nn.functional as F


def solve(k, q_block, bef_filtered_len, block_len, keep_ratio=0.5, pool_kernel_size=3):
    """Sparse-dLLM 的「延迟一步 + 双向块视图」KV 驱逐（float64 内部，按输入 dtype 返回）。"""
    for name, tensor in (("k", k), ("q_block", q_block)):
        if tensor.dim() != 4:
            raise ValueError(f"{name} 必须是四维")
        if tensor.dtype not in (torch.float32, torch.float64):
            raise ValueError(f"{name} 必须是 float32/float64")
    if k.dtype != q_block.dtype:
        raise ValueError("k 与 q_block 必须同 dtype")
    batch, kv_heads, kv_len, head_dim = k.shape
    if q_block.shape[0] != batch or q_block.shape[3] != head_dim:
        raise ValueError("batch 与 head_dim 必须一致")
    q_heads = int(q_block.shape[1])
    if q_heads % kv_heads:
        raise ValueError("query head 数必须是 kv head 数的整数倍（GQA）")
    for name, value in (("bef_filtered_len", bef_filtered_len), ("block_len", block_len),
                        ("pool_kernel_size", pool_kernel_size)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} 必须是 int")
    if block_len < 1 or bef_filtered_len < 0:
        raise ValueError("block_len >= 1 且 bef_filtered_len >= 0")
    if block_len != q_block.shape[2]:
        raise ValueError("block_len 必须等于 q_block 的块长度")
    if isinstance(keep_ratio, bool) or not isinstance(keep_ratio, (int, float)):
        raise ValueError("keep_ratio 必须是浮点数")
    if not 0.0 < float(keep_ratio) <= 1.0:
        raise ValueError("keep_ratio 必须落在 (0, 1]")
    if pool_kernel_size < 1 or pool_kernel_size % 2 == 0:
        raise ValueError("pool_kernel_size 必须是正奇数")
    if bef_filtered_len + block_len > kv_len:
        raise ValueError("块窗口越界：bef_filtered_len + block_len 不能超过缓存长度")

    out_dtype = k.dtype
    # ① 延迟一步：先把当前块整段剔出候选，再与前后两段拼回去
    cand = torch.cat([k[:, :, :bef_filtered_len], k[:, :, bef_filtered_len + block_len:]], dim=2)
    if cand.shape[2] < 1:
        raise ValueError("块外没有任何候选 token")
    cand = cand.to(torch.float64)
    if q_heads != kv_heads:                                   # ③ GQA：KV head 按组复制到 query head 上
        cand = cand.repeat_interleave(q_heads // kv_heads, dim=1)
    q64 = q_block.to(torch.float64)
    avg_q = q64.mean(dim=-2)                                  # ② 块内 query 取均值（双向，无因果）
    scores = (avg_q.unsqueeze(-2) @ cand.transpose(-2, -1)).squeeze(-2)   # (B, Hq, S')
    heads_avg = scores.mean(dim=1)                            # ③ 先跨 head 平均
    pooled = heads_avg
    if pool_kernel_size > 1:                                  # ④ kernel 3 / stride 1 / pad 1（-inf 填充）
        pooled = F.max_pool1d(heads_avg.unsqueeze(1), pool_kernel_size, 1,
                              pool_kernel_size // 2).squeeze(1)
    keep_num = int(pooled.shape[-1] * keep_ratio)             # ⑤ 截断取整，可为 0
    keep_idx = torch.argsort(-pooled, dim=-1, stable=True)[:, :keep_num].to(torch.int64)

    return {
        "avg_q": avg_q.to(out_dtype),
        "scores": scores.to(out_dtype),
        "heads_avg": heads_avg.to(out_dtype),
        "pooled": pooled.to(out_dtype),
        "keep_idx": keep_idx.contiguous(),
    }
