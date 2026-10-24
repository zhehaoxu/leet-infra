import torch


def solve(logits, k, u):
    # TODO: Top-k 采样（u 为显式输入，保证确定性）。
    # probs=softmax(logits) → 保留 logit 最大的 k 个（其余置0）→ 重新归一化得 filtered →
    # 用 u 在 filtered 原顺序 CDF 上逆变换采样得 chosen。返回 (filtered, chosen)。
    raise NotImplementedError
