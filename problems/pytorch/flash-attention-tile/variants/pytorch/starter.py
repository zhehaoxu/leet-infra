import torch


def solve(q, k, v, block=2):
    # TODO: 用 FlashAttention 的分块在线 softmax 计算注意力：
    # 对 query 分块、每块沿 key 维分块遍历，维护 running max m / 分母 l / 分子 acc，
    # 遇到更大 m 时用 alpha=exp(m_old-m_new) rescale。缩放 1/sqrt(d)，无掩码。
    # 结果须与标准 softmax(q k^T / sqrt(d)) v 数值等价（对任意 block、非整除长度都成立）。
    raise NotImplementedError
