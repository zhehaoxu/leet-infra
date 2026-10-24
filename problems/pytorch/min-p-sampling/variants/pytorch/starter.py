import torch


def solve(logits, p, u):
    # TODO: Min-p 采样（u 为显式输入，保证确定性）。
    # probs=softmax(logits) → 保留 probs >= p*probs.max() → 归一化得 filtered →
    # 用 u 在 filtered 原顺序 CDF 上逆变换采样得 chosen。返回 (filtered, chosen)。
    raise NotImplementedError
