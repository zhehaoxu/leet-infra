import torch


def solve(logits, u, tau=1.0):
    # TODO: Gumbel-Softmax。g = -log(-log(u))（Gumbel 噪声），
    # 输出 softmax((logits + g) / tau)，沿最后一维。u 为显式输入的 (0,1) 均匀随机数。
    raise NotImplementedError
