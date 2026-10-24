"""参考答案：GRU 单元 GRUCell 单步前向。

两次全量矩阵乘得 gi/gh（各 (B, 3H)），按 (reset, update, new) 切三段，
r、z 走 sigmoid、候选 n 走 tanh 且 r 只乘 gh 的 new 段，最后 (1-z)*n + z*h。
与 nn.GRUCell 权重布局一致。
"""

import torch


def solve(x, h, w_ih, w_hh, b_ih, b_hh):
    H = h.shape[1]
    gi = x @ w_ih.T + b_ih
    gh = h @ w_hh.T + b_hh
    ri, zi, ni = gi.split(H, 1)
    rh, zh, nh = gh.split(H, 1)
    r = torch.sigmoid(ri + rh)
    z = torch.sigmoid(zi + zh)
    n = torch.tanh(ni + r * nh)
    return (1 - z) * n + z * h
