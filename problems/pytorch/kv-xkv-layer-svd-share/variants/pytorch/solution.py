import torch


def solve(k, layer_group_size=2, rank_k=256):
    """xKV 跨层共享奇异向量：分组拼通道轴做一次 SVD，U 共享、每层只留 S·V（float64 内部）。"""
    if k.dim() != 4:
        raise ValueError("k 必须是四维 (num_layers, S, H, D)")
    if k.dtype not in (torch.float32, torch.float64):
        raise ValueError("k 必须是 float32/float64")
    num_layers, seq, heads, dim = (int(s) for s in k.shape)
    if min(num_layers, seq, heads, dim) < 1:
        raise ValueError("每一维都必须 >= 1")
    for name, value in (("layer_group_size", layer_group_size), ("rank_k", rank_k)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} 必须是 int")
        if value < 1:
            raise ValueError(f"{name} 必须 >= 1")
    if num_layers % layer_group_size:
        raise ValueError("层数必须被 layer_group_size 整除")

    out_dtype = k.dtype
    k64 = k.to(torch.float64)
    chan = heads * dim                                   # 每层通道数
    width = layer_group_size * chan                       # 组内拼接后的通道数
    rank = min(rank_k, seq, width)                        # 秩按拼接后的宽度截断
    groups = num_layers // layer_group_size
    rows = torch.arange(rank)

    us, coefs, recons, errs = [], [], [], []
    for g in range(groups):
        lo = g * layer_group_size
        # (gs, S, H, D) -> (S, gs*H*D)：行是位置，列是「层在前、层内 head-major」的拼接通道
        mat = k64[lo: lo + layer_group_size].permute(1, 0, 2, 3).reshape(seq, width)
        u, s, vh = torch.linalg.svd(mat, full_matrices=False)
        u, sv = u[:, :rank], torch.diag(s[:rank]) @ vh[:rank]      # SV = S·V，形状 (rank, width)
        # 确定性符号：u 每列翻转成「绝对值最大的分量为正」，并列时取最小行号
        peak = u.abs().max(dim=0, keepdim=True).values
        pivot = (u.abs() == peak).to(torch.int64).argmax(dim=0)
        flip = torch.where(u[pivot, rows] < 0, -1.0, 1.0)
        u, sv = u * flip, sv * flip.unsqueeze(1)

        recon = u @ sv                                          # (S, width)
        us.append(u)
        recons.append(recon.reshape(seq, layer_group_size, heads, dim).permute(1, 0, 2, 3))
        # (rank, gs, H, D) -> 每层一份 (H, rank, D)
        coefs.append(sv.reshape(rank, layer_group_size, heads, dim).permute(1, 2, 0, 3))
        errs.append((mat - recon).norm())

    return {
        "u": torch.stack(us).to(out_dtype),
        "coef": torch.cat(coefs, dim=0).to(out_dtype),
        "recon": torch.cat(recons, dim=0).to(out_dtype),
        "err": torch.stack(errs).to(out_dtype),
    }
