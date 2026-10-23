import torch


def solve(q, k, v, window):
    # TODO: 因果 + 滑动窗口注意力。位置 i 只能关注 max(0, i-window+1) <= j <= i
    # （即 j <= i 且 i - j < window）。缩放 1/sqrt(d)，q/k/v 形状 (B, L, d)。
    # 对角线恒可见，window >= L 时退化为纯因果注意力。
    raise NotImplementedError
