import torch

_POS_LR = (0.01, 0.02, 0.04, 0.08, 0.12, 0.16)
_NEG_LR = (-0.01, -0.02, -0.04)


def _flat_cos(a, b):
    a = a.flatten()
    b = b.flatten()
    a = a / (a.norm() + 1e-8)
    b = b / (b.norm() + 1e-8)
    return float(torch.clamp((a * b).sum(), -1.0, 1.0))


def _row_cos(a, b):
    a = a / (a.norm(dim=1, keepdim=True) + 1e-8)
    b = b / (b.norm(dim=1, keepdim=True) + 1e-8)
    return torch.clamp((a * b).sum(dim=1), -1.0, 1.0)


def _prune(W, sorted_idx, sparsities):
    rows = torch.arange(W.shape[0]).unsqueeze(1).expand(-1, W.shape[1])
    cols = torch.arange(W.shape[1]).unsqueeze(0).expand(W.shape[0], -1)
    num_prune = (W.shape[1] * sparsities).long()
    prune_mask = cols < num_prune.unsqueeze(1)
    keep = torch.ones_like(W, dtype=torch.bool)
    keep[rows, sorted_idx] = ~prune_mask
    return W * keep, keep


def solve(weight, inputs, activation_ms, target_sparsity, iterations=10,
          qmetric="cosim_flat", qmetric_dimwise="cosim", forbid_neg_lr=False,
          lr_grid=_POS_LR, neg_lr_grid=_NEG_LR):
    """TRIM：逐行稀疏率按输出余弦反馈迭代（2505.16743，flobk/TRIM 口径）。"""
    if weight.dim() != 2:
        raise ValueError("weight 必须是二维 (out, in)")
    if weight.shape[0] < 2:
        raise ValueError("out_features 必须 >= 2（逐行方差的定义要求）")
    if inputs.dim() != 2:
        raise ValueError("inputs 必须是二维 (in, tokens)")
    if activation_ms.dim() != 1:
        raise ValueError("activation_ms 必须是一维")
    if activation_ms.shape[0] != weight.shape[1]:
        raise ValueError("activation_ms 长度必须等于 in_features")
    if inputs.shape[0] != weight.shape[1]:
        raise ValueError("inputs 的行数必须等于 in_features")
    if inputs.shape[1] < 1:
        raise ValueError("inputs 至少需要 1 个 token")
    if not (weight.dtype == inputs.dtype == activation_ms.dtype
            and weight.dtype in (torch.float32, torch.float64)):
        raise ValueError("weight/inputs/activation_ms 必须同 dtype 且为 float32/float64")
    if isinstance(target_sparsity, bool) or not isinstance(target_sparsity, (int, float)):
        raise ValueError("target_sparsity 必须是数值")
    if not 0.0 < float(target_sparsity) < 1.0:
        raise ValueError("target_sparsity 必须落在 (0, 1)")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 0:
        raise ValueError("iterations 必须是非负 int")
    if qmetric != "cosim_flat" or qmetric_dimwise != "cosim":
        raise ValueError("本题只支持 qmetric='cosim_flat'、qmetric_dimwise='cosim'")
    for name, grid in (("lr_grid", lr_grid), ("neg_lr_grid", neg_lr_grid)):
        if not isinstance(grid, (list, tuple)):
            raise ValueError(f"{name} 必须是序列")

    dtype = weight.dtype
    W = weight.to(torch.float64)
    X = inputs.to(torch.float64)
    act = activation_ms.to(torch.float64)
    rms = act.sqrt().unsqueeze(0)
    sorted_idx = torch.sort(W.abs() * rms, dim=-1, stable=True)[1]
    unpruned = W @ X
    target = float(target_sparsity)

    def run_iterative(learning_rate):
        x = torch.full((W.shape[0],), target, dtype=torch.float64)
        pruned = _prune(W, sorted_idx, x)[0] @ X
        best_metric = _flat_cos(unpruned, pruned)
        best_x = x.clone()
        best_iter = 0
        for step in range(iterations):
            qm = _row_cos(unpruned, pruned)
            delta = (qm - qm.min()) / (qm.max() - qm.min() + 1e-6)
            delta = delta * (learning_rate * 2.0)
            delta = delta - delta.mean()
            x = x + delta
            x = torch.clamp(x, max=0.95)
            x = torch.clamp(x, min=0.0)
            x = x - x.mean() + target
            pruned = _prune(W, sorted_idx, x)[0] @ X
            score = _flat_cos(unpruned, pruned)
            if score > best_metric:
                best_metric = score
                best_x = x.clone()
                best_iter = step
        return best_x, best_metric, best_iter

    pruned0 = _prune(W, sorted_idx, torch.full((W.shape[0],), target,
                                               dtype=torch.float64))[0] @ X
    baseline_metric = _flat_cos(unpruned, pruned0)
    starting_var = float(_row_cos(unpruned, pruned0).var())
    best_metric = baseline_metric
    best_sparsities = torch.full((W.shape[0],), target, dtype=torch.float64)
    best_learning_rate = None
    best_iter = 0
    best_this_method = float("-inf")
    for learning_rate in lr_grid:
        sparsities, score, step = run_iterative(float(learning_rate))
        if score > best_metric:
            best_metric = score
            best_sparsities = sparsities
            best_learning_rate = float(learning_rate)
            best_iter = step
        if score > best_this_method:
            best_this_method = score
        else:
            break
    if best_learning_rate is None and not forbid_neg_lr:
        best_this_method = float("-inf")
        for learning_rate in neg_lr_grid:
            sparsities, score, step = run_iterative(float(learning_rate))
            if score > best_metric:
                best_metric = score
                best_sparsities = sparsities
                best_learning_rate = float(learning_rate)
                best_iter = step
            if score > best_this_method:
                best_this_method = score
            else:
                break

    ending_var = float(_row_cos(unpruned, _prune(W, sorted_idx, best_sparsities)[0] @ X).var())

    final = best_sparsities.clone()
    final = torch.clamp(final, max=0.95)
    final = torch.clamp(final, min=0.0)
    final = final - final.mean() + target
    pruned_weight, keep_mask = _prune(W, sorted_idx, final)

    return {
        "sparsities": final,
        "keep_mask": keep_mask,
        "pruned_weight": pruned_weight.to(dtype),
        "baseline_metric": float(baseline_metric),
        "best_metric": float(best_metric),
        "learning_rate": best_learning_rate,
        "best_iter": int(best_iter),
        "starting_var": starting_var,
        "ending_var": ending_var,
    }
