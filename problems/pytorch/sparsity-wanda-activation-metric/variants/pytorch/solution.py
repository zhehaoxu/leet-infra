import torch


def solve(weight, activations, sparsity, prune_n=0, prune_m=0):
    """Wanda：|W|·RMS(激活) 逐行掩码；2:4 时按 m 列分块取最小的 n 个。"""
    if weight.dim() != 2:
        raise ValueError("weight 必须是二维 (out, in)")
    if activations.dim() != 2:
        raise ValueError("activations 必须是二维 (tokens, in)")
    if weight.dtype != activations.dtype or weight.dtype not in (torch.float32, torch.float64):
        raise ValueError("weight/activations 必须同 dtype 且为 float32/float64")
    if activations.shape[0] < 1:
        raise ValueError("activations 至少需要 1 个 token")
    if activations.shape[1] != weight.shape[1]:
        raise ValueError("activations 的列数必须等于 weight 的 in_features")
    if isinstance(sparsity, bool) or not isinstance(sparsity, (int, float)):
        raise ValueError("sparsity 必须是数值")
    if not 0.0 <= float(sparsity) <= 1.0:
        raise ValueError("sparsity 必须落在 [0, 1]")
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
            raise ValueError("N:M 模式下 sparsity 必须等于 0.5（上游 main.py 的断言）")

    w64 = weight.to(torch.float64)
    col_ms = activations.to(torch.float64).pow(2).mean(dim=0)
    metric = w64.abs() * col_ms.sqrt().unsqueeze(0)

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
        "mask": mask,
        "pruned_weight": pruned_weight.to(weight.dtype),
        "sparsity": float(mask.sum().item()) / float(mask.numel()),
    }
