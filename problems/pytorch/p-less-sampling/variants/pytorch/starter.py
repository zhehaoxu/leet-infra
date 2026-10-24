import torch


def solve(logits, u):
    # TODO: P-less 采样（免超参；u 为显式输入，保证确定性）。
    # probs=softmax → thr=(probs**2).sum(-1,keepdim) → 保留 probs >= thr →
    # 保留集合归一化得 filtered → 用 u 在 filtered CDF 逆变换采样得 chosen。
    # 返回 (filtered, chosen)。
    raise NotImplementedError
