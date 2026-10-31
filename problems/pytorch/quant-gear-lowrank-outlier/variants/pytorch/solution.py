"""参考解：GEAR 稀疏离群 + 分组非对称量化 + 残差低秩幂迭代补偿。

compress_function.py::gears_tokenQ L300-333（离群名额/均值替换/写回）
compress_function.py::fake_groupwise_token_asymmetric_quantization_cluster L132-161
compress_function.py::fake_poweriteration_group L69-98（QR 只在最后一轮）
compress_function.py::gearslkivi_tokenQ_new L204-211（output + low_rank）
"""

import torch


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def solve(kv, bits, group_size, sparsity, rank, loop, p_base, q_base):
    if not torch.is_tensor(kv) or kv.dim() != 4:
        raise ValueError("kv 必须是四维 (n_batch, n_head, seq_len, head_dim)")
    if kv.dtype not in (torch.float32, torch.float64):
        raise ValueError("kv 必须是 float32/float64")
    for name, val, lo in (("bits", bits, 1), ("group_size", group_size, 1),
                          ("rank", rank, 1), ("loop", loop, 1)):
        if not _is_int(val) or val < lo:
            raise ValueError(f"{name} 必须是 >= {lo} 的整数")
    if isinstance(sparsity, bool) or not isinstance(sparsity, (int, float)) \
            or not 0.0 <= sparsity < 1.0:
        raise ValueError("sparsity 必须落在 [0, 1)")
    batch, num_head, seq_len, head_dim = kv.shape
    channels = num_head * head_dim
    if channels % group_size:
        raise ValueError("group_size 必须整除 n_head*head_dim")
    if rank > min(seq_len, head_dim):
        raise ValueError("rank 不能超过 seq_len 与 head_dim")
    for name, base, shape in (("p_base", p_base, (batch, num_head, head_dim, rank)),
                              ("q_base", q_base, (batch, num_head, seq_len, rank))):
        if not torch.is_tensor(base) or tuple(base.shape) != shape:
            raise ValueError(f"{name} 形状应为 {shape}")
        if base.dtype != kv.dtype:
            raise ValueError(f"{name} 的 dtype 必须与 kv 一致")

    out_dtype = kv.dtype
    merged = kv.to(torch.float64).permute(0, 2, 1, 3).reshape(batch, seq_len, channels)

    n_out = int(int(batch * channels * seq_len * sparsity) / batch / seq_len / 2)
    small_idx = torch.argsort(merged, dim=-1, stable=True)[..., :n_out]
    large_idx = torch.argsort(-merged, dim=-1, stable=True)[..., :n_out]
    row_mean = merged.mean(dim=-1, keepdim=True)
    filled = merged.clone()
    for idx in (small_idx, large_idx):
        filled.scatter_(-1, idx, row_mean.expand(-1, -1, idx.shape[-1]))

    groups = filled.reshape(batch, seq_len, channels // group_size, group_size)
    mx = groups.max(dim=-1, keepdim=True).values
    mn = groups.min(dim=-1, keepdim=True).values
    scale = (mx - mn) / (2 ** bits - 1)
    code = ((groups - mn) / torch.where(scale > 0, scale, torch.ones_like(scale))).clamp(min=0).round_()
    base_q = (code * scale + mn).reshape(batch, seq_len, channels)
    for idx in (small_idx, large_idx):
        base_q.scatter_(-1, idx, merged.gather(-1, idx))

    def split(t):
        return t.reshape(batch, seq_len, num_head, head_dim).permute(0, 2, 1, 3).contiguous()

    quantized = split(base_q)
    residual = kv.to(torch.float64) - quantized

    p, q = p_base.to(torch.float64), q_base.to(torch.float64)
    a = residual
    for i in range(loop):
        if i == loop - 1:
            p = torch.linalg.qr(p).Q
        q = a @ p
        if i == loop - 1:
            q = torch.linalg.qr(q).Q
        p = a.transpose(-1, -2) @ q
    low_rank = q @ p.transpose(-1, -2)

    mask = torch.zeros(batch, seq_len, channels, dtype=torch.bool)
    mask.scatter_(-1, small_idx, True)
    mask.scatter_(-1, large_idx, True)

    def cast(t):
        return t.to(out_dtype) if out_dtype == torch.float32 else t

    return {"outlier_mask": split(mask), "sparse_filled": cast(split(filled)),
            "quantized": cast(quantized), "residual": cast(residual),
            "low_rank": cast(low_rank), "output": cast(quantized + low_rank)}
