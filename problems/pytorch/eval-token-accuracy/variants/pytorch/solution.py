"""参考答案：token 级准确率，同时给出 micro（全局）与 macro（逐样本再平均）两个口径。"""

import torch


def solve(pred_ids, labels, ignore_index=-100):
    valid = labels != ignore_index
    hit = (pred_ids == labels) & valid
    total_valid = int(valid.sum().item())
    if total_valid == 0:
        return 0.0, 0.0
    micro = int(hit.sum().item()) / total_valid          # Python float64 相除
    per_valid = valid.sum(dim=1)
    per_hit = hit.sum(dim=1)
    per_acc = per_hit.double() / per_valid.double().clamp(min=1)
    sel = per_valid > 0
    macro = float(per_acc[sel].mean().item()) if int(sel.sum().item()) > 0 else 0.0
    return micro, macro
