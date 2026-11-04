"""LLM-Barber 的 mask_rebuild：对已剪枝稀疏权重做 mask 重建。

给定稠密权重、稀疏参考权重（0 位即当前 mask）与梯度² Fisher 量（scaler，
由调用方显式提供——上游通过 block 前向 MSE 反传的 grad² 滑动平均采集，
不在本题判题域），按三种度量（WD = |W|·√scaler、W = |W|、D = √scaler）
计算 metric，再用 find_threshold 求 growth/prune 双阈值：growth 集合
（当前被剪、metric 高）与 prune 集合（当前存活、metric 低）按
recons_num = Σ(growth 降序前 min_len 项 > prune 升序前 min_len 项) 配对，
阈值取 int(recons_num·threshold) 处（越界取末元素；空集合取 +inf/−inf），
最后交换：growth 位恢复（mask 置 False）、prune 位剪掉（mask 置 True），
重建权重并把 mask 位清零。

粒度四选：block（全部矩阵拼接后全局一对阈值）、layer（逐矩阵）、
output1（逐输出行）、input1（逐输入列）。retain_reconstruction 决定
combined 权重是 mask 位回填稠密值还是整体取稠密权重。
"""

import math

import torch

_METRICS = ("WD", "W", "D")
_GRANULARITIES = ("block", "layer", "output1", "input1")


def _validate_dict(d, name):
    if not isinstance(d, dict) or not d:
        raise ValueError(f"{name} 必须是非空 dict")
    for key, value in d.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} 的键必须是非空字符串")
        if not isinstance(value, torch.Tensor) or value.dim() != 2:
            raise ValueError(f"{name}[{key}] 必须是 2 维 (H, W) 张量")
        if not torch.is_floating_point(value):
            raise ValueError(f"{name}[{key}] 必须是浮点张量")


def _check_shapes(weights, ref_weights, scalers):
    if set(weights) != set(ref_weights) or set(weights) != set(scalers):
        raise ValueError("weights / ref_weights / scalers 的键必须一致")
    for name in weights:
        shape = weights[name].shape
        if ref_weights[name].shape != shape:
            raise ValueError(f"ref_weights[{name}] 的形状必须与 weights[{name}] 一致")
        if scalers[name].shape != shape:
            raise ValueError(f"scalers[{name}] 的形状必须与 weights[{name}] 一致")
        if (scalers[name] < 0).any():
            raise ValueError(f"scalers[{name}] 必须非负（Fisher 量是 grad² 的滑动平均）")


def _find_threshold(growth_metric, prune_metric, threshold):
    """官方 find_threshold：growth 降序 vs prune 升序配对计数定双阈值。"""
    if growth_metric.numel() == 0 or prune_metric.numel() == 0:
        return float("inf"), float("-inf")
    growth_sort = torch.sort(growth_metric, descending=True).values
    prune_sort = torch.sort(prune_metric, descending=False).values
    min_len = min(growth_sort.numel(), prune_sort.numel())
    recons_num = int(torch.sum(
        (growth_sort[:min_len] - prune_sort[:min_len]) > 0).item())
    idx = int(recons_num * threshold)
    growth_threshold = growth_sort[-1].item() if idx == growth_sort.numel() \
        else growth_sort[idx].item()
    prune_threshold = prune_sort[-1].item() if idx == prune_sort.numel() \
        else prune_sort[idx].item()
    return growth_threshold, prune_threshold


def _metric_of(combined, scaler, metric):
    if metric == "WD":
        return combined.abs() * scaler.sqrt()
    if metric == "W":
        return combined.abs()
    return scaler.sqrt()  # "D"


def solve(weights, ref_weights, scalers, metric="WD", threshold=1.0,
          granularity="layer", retain_reconstruction=True):
    _validate_dict(weights, "weights")
    _validate_dict(ref_weights, "ref_weights")
    _validate_dict(scalers, "scalers")
    _check_shapes(weights, ref_weights, scalers)
    if metric not in _METRICS:
        raise ValueError(f"metric 必须是 {'/'.join(_METRICS)} 之一")
    if granularity not in _GRANULARITIES:
        raise ValueError(f"granularity 必须是 {'/'.join(_GRANULARITIES)} 之一")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) \
            or not math.isfinite(threshold) or not (0.0 <= threshold <= 1.0):
        raise ValueError("threshold 必须是 [0, 1] 内的有限数值")

    masks, metrics, combined = {}, {}, {}
    for name in weights:
        mask = ref_weights[name] == 0
        if retain_reconstruction:
            comb = torch.where(mask, weights[name], ref_weights[name])
        else:
            comb = weights[name].clone()
        combined[name] = comb.double()
        masks[name] = mask
        metrics[name] = _metric_of(combined[name], scalers[name].double(), metric)

    if granularity == "block":
        growth_cat = torch.cat(
            [metrics[name][masks[name]] for name in weights])
        prune_cat = torch.cat(
            [metrics[name][~masks[name]] for name in weights])
        block_gt, block_pt = _find_threshold(growth_cat, prune_cat, threshold)

    thresholds = {}
    rebuilt = {}
    zero_cnt, total_cnt = 0, 0
    for name in weights:
        m = metrics[name]
        mask = masks[name]
        if granularity == "block":
            gt, pt = block_gt, block_pt
        elif granularity == "layer":
            gt, pt = _find_threshold(m[mask], m[~mask], threshold)
        elif granularity == "output1":
            gt = [float("inf")] * m.shape[0]
            pt = [float("-inf")] * m.shape[0]
            for j in range(m.shape[0]):
                gt[j], pt[j] = _find_threshold(m[j][mask[j]], m[j][~mask[j]],
                                               threshold)
        else:  # input1
            gt = [float("inf")] * m.shape[1]
            pt = [float("-inf")] * m.shape[1]
            for j in range(m.shape[1]):
                gt[j], pt[j] = _find_threshold(m[:, j][mask[:, j]],
                                               m[:, j][~mask[:, j]], threshold)

        if granularity in ("output1", "input1"):
            axis = 1 if granularity == "output1" else 0
            growth_indices = m > torch.tensor(gt, dtype=m.dtype).unsqueeze(axis)
            prune_indices = m < torch.tensor(pt, dtype=m.dtype).unsqueeze(axis)
        else:
            growth_indices = m > gt
            prune_indices = m < pt

        mask = mask.clone()
        mask[growth_indices] = False
        mask[prune_indices] = True
        masks[name] = mask
        thresholds[name] = (gt, pt)
        rebuilt[name] = torch.where(mask, torch.zeros_like(combined[name]),
                                    combined[name])
        zero_cnt += int(mask.sum().item())
        total_cnt += mask.numel()

    return {
        "masks": masks,
        "rebuilt": rebuilt,
        "sparsity": zero_cnt / total_cnt,
        "thresholds": thresholds,
    }
