import torch


def solve(weight, activations, grad_samples, sparsity, scale=1.0,
          gradient_inv=False, prune_n=0, prune_m=0):
    """GBLM：|W|·√s_row ⊕ 梯度项 的融合显著性逐行掩码。

    上游 VILA-Lab/GBLM-Pruner 的 prune_gblm（lib/prune.py）：
      scaler_row  = Σ_tokens x²（WrappedGPT 口径，本题沿用 581 题的 mean 惯例，
                    常数因子不影响排序，metric 值按 mean 口径对拍）
      G           = sqrt( Σ_samples (g_s · scale)² )   —— L2 聚合
      非 inv      : metric = |W|·sqrt(scaler_row) + |W|·G
      inv         : metric = |W|·sqrt(scaler_row) · 1/(G + 1e-8)
    非结构化：每行 stable sort 取前 int(in·sparsity) 个最小置零；
    N:M：每 m 列块取最小 n 个。
    """
    if weight.dim() != 2:
        raise ValueError("weight 必须是二维 (out, in)")
    if activations.dim() != 2:
        raise ValueError("activations 必须是二维 (tokens, in)")
    if weight.dtype != activations.dtype or weight.dtype not in (torch.float32, torch.float64):
        raise ValueError("weight/activations 必须同 dtype 且为 float32/float64")
    if activations.shape[1] != weight.shape[1]:
        raise ValueError("activations 的列数必须等于 weight 的 in_features")
    if grad_samples.dim() != 3:
        raise ValueError("grad_samples 必须是三维 (samples, out, in)")
    if grad_samples.shape[0] < 1:
        raise ValueError("grad_samples 至少需要 1 个样本")
    if grad_samples.shape[1:] != weight.shape:
        raise ValueError("grad_samples 每个样本的形状必须与 weight 相同")
    if grad_samples.dtype not in (torch.float32, torch.float64):
        raise ValueError("grad_samples 必须是 float32/float64")
    if isinstance(sparsity, bool) or not isinstance(sparsity, (int, float)):
        raise ValueError("sparsity 必须是数值")
    if not 0.0 <= float(sparsity) <= 1.0:
        raise ValueError("sparsity 必须落在 [0, 1]")
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        raise ValueError("scale 必须是数值")
    if float(scale) <= 0:
        raise ValueError("scale 必须为正")
    if not isinstance(gradient_inv, bool):
        raise ValueError("gradient_inv 必须是布尔值")
    for name, value in (("prune_n", prune_n), ("prune_m", prune_m)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} 必须是 int")
        if value < 0:
            raise ValueError(f"{name} 必须 >= 0")

    out_features, in_features = weight.shape
    structured = prune_n > 0
    if structured:
        if prune_m <= 0 or prune_n >= prune_m:
            raise ValueError("N:M 需要 0 < prune_n < prune_m")
        if in_features % prune_m != 0:
            raise ValueError("in_features 必须能被 prune_m 整除")
        if float(sparsity) != 0.5:
            raise ValueError("N:M 模式下 sparsity 必须等于 0.5")

    w64 = weight.to(torch.float64)
    # scaler_row：激活列平方和（mean 口径，与 581 题一致）
    col_ms = activations.to(torch.float64).pow(2).mean(dim=0)

    # 梯度 L2 聚合：G = sqrt(Σ (g_s·scale)²)
    scaled = grad_samples.to(torch.float64) * float(scale)
    grad_l2 = scaled.pow(2).sum(dim=0).sqrt()

    base = w64.abs() * col_ms.sqrt().unsqueeze(0)
    if not gradient_inv:
        metric = base + w64.abs() * grad_l2
    else:
        metric = base / (grad_l2 + 1e-8)

    mask = torch.zeros_like(metric, dtype=torch.bool)
    if structured:
        for start in range(0, in_features, prune_m):
            block = metric[:, start:start + prune_m]
            order = torch.sort(block, dim=-1, stable=True)[1][:, :prune_n]
            mask.scatter_(1, order + start, True)
    else:
        keep_pruned = int(in_features * float(sparsity))
        order = torch.sort(metric, dim=-1, stable=True)[1][:, :keep_pruned]
        mask.scatter_(1, order, True)

    pruned_weight = w64.masked_fill(mask, 0.0)
    return {
        "metric": metric.to(weight.dtype),
        "grad_l2": grad_l2.to(weight.dtype),
        "mask": mask,
        "pruned_weight": pruned_weight.to(weight.dtype),
        "sparsity": float(mask.sum().item()) / float(mask.numel()),
    }
