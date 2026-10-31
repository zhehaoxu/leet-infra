"""参考解：vqllm 的 std 归一化 + 多码本多步残差向量量化（含 key 侧通道重排）。

对照 iankur/vqllm（arXiv 2410.15704）：
  quantizer/utils.py:23-25  scale = x.std(-1, keepdim=True)，0 换成 1，x /= scale
  quantizer/utils.py:44-52  逐 quantizer 逐轮：xq = quantizer(x)、x = x - xq
  quantizer/utils.py:55     x_orig - 累积残差；L58 x *= scale
  quantizer/utils.py:71-78  view/transpose/reshape 重排 + split/stack 复原
  quantizer/quantize.py:146-181  z.view(-1,N,d).permute → -(||z||²+||c||²-2cz) → argmax → 加 k*M 偏移
"""

import torch


def solve(x, codebooks, num_residual_steps=1, reorder_channel=False):
    """vqllm 残差向量量化：std 归一化 → 逐量化器逐轮最近邻 → 残差累减 → 尺度还原。"""
    if not isinstance(x, torch.Tensor) or x.dim() != 3:
        raise ValueError("x 必须是三维 (B, L, D)")
    if x.dtype not in (torch.float32, torch.float64):
        raise ValueError("x 必须是 float32/float64")
    if not isinstance(reorder_channel, bool):
        raise ValueError("reorder_channel 必须是 bool")
    if isinstance(num_residual_steps, bool) or not isinstance(num_residual_steps, int) \
            or num_residual_steps < 1:
        raise ValueError("num_residual_steps 必须是 >= 1 的 int")
    if not isinstance(codebooks, (list, tuple)) or len(codebooks) == 0:
        raise ValueError("codebooks 必须是非空 list/tuple")
    shape = tuple(int(v) for v in x.shape)
    dim = shape[-1]
    if dim < 2:
        raise ValueError("D 必须 >= 2：std 用 correction=1，D=1 会得到 NaN")
    head = codebooks[0]
    if not isinstance(head, torch.Tensor) or head.dim() != 3:
        raise ValueError("每个 codebook 必须是三维 (N, M, dhat)")
    n_cb, dhat = int(head.shape[0]), int(head.shape[2])
    if n_cb < 1 or dhat < 1 or n_cb * dhat != dim:
        raise ValueError("必须满足 D == num_codebooks * codebook_entry_dim")
    for cb in codebooks:
        if not isinstance(cb, torch.Tensor) or cb.dim() != 3 or cb.dtype != x.dtype:
            raise ValueError("每个 codebook 必须是三维 (N, M, dhat) 且与 x 同 dtype")
        if int(cb.shape[0]) != n_cb or int(cb.shape[2]) != dhat or int(cb.shape[1]) < 1:
            raise ValueError("所有 codebook 共享 (N, dhat)，且 M >= 1")

    out_dtype = x.dtype
    work = x.detach().to(torch.float64)
    if reorder_channel:                                        # utils.py:71-72
        work = work.reshape(*shape[:-1], n_cb, dhat).transpose(-1, -2).reshape(*shape)

    scale = work.std(dim=-1, keepdim=True)                     # utils.py:23，correction=1
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    z = work / scale
    residual = z.clone()

    norms, indices = [], []
    for cb in codebooks:                                       # utils.py:44
        table = cb.detach().to(torch.float64)
        n_entries = int(table.shape[1])
        offs = torch.arange(n_entries, dtype=torch.int64)
        trail = [torch.linalg.vector_norm(residual, dim=-1)]
        picks = []
        for _ in range(num_residual_steps):                     # utils.py:48
            seg = residual.reshape(*shape[:-1], n_cb, dhat)
            chosen, step = [], []
            for k in range(n_cb):
                s = seg[..., k, :]
                dist = (s * s).sum(-1, keepdim=True) + (table[k] * table[k]).sum(-1)
                dist = dist - 2.0 * (s @ table[k].transpose(0, 1))   # quantize.py:167-171
                hit = dist == dist.min(dim=-1, keepdim=True).values
                idx = torch.where(hit, offs, torch.full((), n_entries)).min(dim=-1).values
                step.append(idx + k * n_entries)                # quantize.py:176-177
                chosen.append(table[k][idx])
            residual = residual - torch.stack(chosen, dim=-2).reshape(*shape)
            picks.append(torch.stack(step, dim=-1))
            trail.append(torch.linalg.vector_norm(residual, dim=-1))
        indices.append(torch.stack(picks, dim=0))
        norms.append(torch.stack(trail, dim=0))

    recon = z - residual                                    # utils.py:55
    if reorder_channel:                                     # utils.py:77-78
        recon = torch.stack(recon.split(dhat, dim=-1), dim=-1).reshape(*shape)
    quantized = recon * scale                               # utils.py:58
    error = torch.linalg.vector_norm(residual, dim=-1) * scale.squeeze(-1)

    return {
        "scale": scale.to(out_dtype),
        "residual_norm": torch.stack(norms, dim=0).to(out_dtype),
        "indices": torch.stack(indices, dim=0),
        "quantized": quantized.to(out_dtype),
        "error": error.to(out_dtype),
    }
