import torch


def solve(weights, activation_ms, sparsity, lamda=0.08, hyper_m=3.0):
    """OWL：逐层离群比 → 归一化密度 → 逐行 Wanda 掩码。"""
    if not isinstance(weights, (list, tuple)) or len(weights) < 2:
        raise ValueError("weights 至少需要 2 层")
    if not isinstance(activation_ms, (list, tuple)) or len(activation_ms) != len(weights):
        raise ValueError("activation_ms 必须与 weights 层数一致")
    if isinstance(sparsity, bool) or not isinstance(sparsity, (int, float)):
        raise ValueError("sparsity 必须是数值")
    if not 0.0 <= float(sparsity) <= 1.0:
        raise ValueError("sparsity 必须落在 [0, 1]")
    if not float(lamda) > 0.0:
        raise ValueError("lamda 必须 > 0")
    if not float(hyper_m) > 0.0:
        raise ValueError("hyper_m 必须 > 0")

    flat_weights = [w for layer in weights for w in layer]
    if not flat_weights:
        raise ValueError("每层至少需要一个权重矩阵")
    dtype = flat_weights[0].dtype
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("权重必须是 float32/float64")

    metric_layers = []
    for layer_index, (layer_w, layer_a) in enumerate(zip(weights, activation_ms)):
        if not layer_w or len(layer_w) != len(layer_a):
            raise ValueError(f"第 {layer_index} 层的权重与激活统计数量不一致或为空")
        layer_metrics = []
        for weight, act_ms in zip(layer_w, layer_a):
            if weight.dim() != 2:
                raise ValueError("权重必须是二维")
            if act_ms.dim() != 1 or act_ms.shape[0] != weight.shape[1]:
                raise ValueError("activation_ms 必须是一维且长度等于 in_features")
            if weight.dtype != dtype or act_ms.dtype != dtype:
                raise ValueError("所有权重与 activation_ms 必须是同一 float dtype")
            layer_metrics.append(weight.to(torch.float64).abs()
                                 * act_ms.to(torch.float64).sqrt().unsqueeze(0))
        metric_layers.append(layer_metrics)

    ratios = []
    for layer_metrics in metric_layers:
        flat = torch.cat([m.flatten() for m in layer_metrics])
        threshold = flat.mean() * float(hyper_m)
        ratios.append(float((flat > threshold).sum().item()) / float(flat.numel()) * 100.0)
    ratios = torch.tensor(ratios, dtype=torch.float64)
    spread = float(ratios.max() - ratios.min())
    if spread == 0.0:
        raise ValueError("各层离群比完全相同，上游的 min-max 归一化会除零")

    normalized = (ratios - ratios.min()) * (2.0 * float(lamda) / spread)
    density = normalized - normalized.mean() + (1.0 - float(sparsity))
    layer_sparsity = 1.0 - density
    layer_sparsity = torch.where(layer_sparsity <= 0.0,
                                 torch.full_like(layer_sparsity, 0.01),
                                 layer_sparsity)

    masks = []
    for layer_metrics, layer_ratio in zip(metric_layers, layer_sparsity.tolist()):
        layer_masks = []
        for metric in layer_metrics:
            in_features = metric.shape[1]
            count = int(in_features * layer_ratio)
            order = torch.sort(metric, dim=-1, stable=True)[1][:, :count]
            mask = torch.zeros_like(metric, dtype=torch.bool)
            mask.scatter_(1, order, True)
            layer_masks.append(mask)
        masks.append(layer_masks)

    metric_out = [[m.to(dtype) for m in layer] for layer in metric_layers]
    pruned = [[w.to(torch.float64).masked_fill(mask, 0.0).to(dtype)
               for w, mask in zip(layer_w, layer_mask)]
              for layer_w, layer_mask in zip(weights, masks)]
    return {
        "outlier_ratio": ratios,
        "density": density,
        "layer_sparsity": layer_sparsity,
        "metric": metric_out,
        "mask": masks,
        "pruned_weight": pruned,
    }
