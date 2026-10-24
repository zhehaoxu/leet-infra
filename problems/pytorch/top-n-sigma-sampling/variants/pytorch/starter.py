import torch


def solve(logits, n, u):
    # TODO: Top-n-sigma 采样（u 为显式输入，保证确定性）。
    # probs=softmax → 逐行算 max/mean/std(总体, 除以 V) → 保留 logit >= max - n*std →
    # 保留集合归一化得 filtered → 用 u 在 filtered CDF 逆变换采样得 chosen。
    # 返回 (filtered, chosen)。
    raise NotImplementedError
