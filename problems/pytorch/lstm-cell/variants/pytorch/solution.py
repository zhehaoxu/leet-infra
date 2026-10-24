"""参考答案：LSTM 单元单步前向（对齐 torch.nn.LSTMCell 的门排布 i, f, g, o）。

约定 gate = x·W_ihᵀ + h·W_hhᵀ + b_ih + b_hh，按 [i, f, g, o] 顺序切成 4 段：
    i = σ(..), f = σ(..), g = tanh(..), o = σ(..)
    c' = f ⊙ c + i ⊙ g
    h' = o ⊙ tanh(c')
"""

import torch
import torch.nn.functional as F


def solve(x, h_prev, c_prev, w_ih, w_hh, b_ih, b_hh):
    gates = F.linear(x, w_ih, b_ih) + F.linear(h_prev, w_hh, b_hh)
    i, f, g, o = gates.chunk(4, dim=1)
    i = torch.sigmoid(i)
    f = torch.sigmoid(f)
    g = torch.tanh(g)
    o = torch.sigmoid(o)
    c_new = f * c_prev + i * g
    h_new = o * torch.tanh(c_new)
    return h_new, c_new
