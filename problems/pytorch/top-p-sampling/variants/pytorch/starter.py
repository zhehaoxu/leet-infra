import torch


def solve(logits, p, u):
    # TODO: Top-p 核采样（u 为显式输入，保证确定性）。
    # softmax→降序 cumsum→保留累积和首次≥p 的前缀→核内归一化(映射回原序, 核外0)→
    # 用 u 对核内 CDF 做逆变换采样。返回 (filtered, chosen_index)。
    raise NotImplementedError
