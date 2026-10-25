import torch


def solve(x, codebook):
    """乘积量化编码：x (n,D) 切成 m 段各 d=D/m 维，
    逐段在第 j 套码本 codebook[j] (k_sub,d) 中取最近中心下标，
    返回 int64 形状 (n, m)，并列取小下标。"""
    # TODO
    raise NotImplementedError
