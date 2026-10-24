import torch


def solve(x, h_prev, c_prev, w_ih, w_hh, b_ih, b_hh):
    # TODO: LSTM 单步前向。gate = x@w_ih.T + h@w_hh.T + b_ih + b_hh，按 [i,f,g,o] 切 4 段。
    # i,f,o=σ(..), g=tanh(..); c' = f*c + i*g; h' = o*tanh(c')。返回 (h_new, c_new)。
    raise NotImplementedError
