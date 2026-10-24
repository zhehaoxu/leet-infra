import torch


def solve(x, h, w_ih, w_hh, b_ih, b_hh):
    # TODO: GRU 单步。x:(B,I), h:(B,H); w_ih:(3H,I), w_hh:(3H,H), b_ih/b_hh:(3H,)。
    # 门块顺序 (reset, update, new)。
    #   gi = x @ w_ih.T + b_ih ; gh = h @ w_hh.T + b_hh
    #   r = sigmoid(gi_r + gh_r); z = sigmoid(gi_z + gh_z)
    #   n = tanh(gi_n + r * gh_n)        # r 只乘 gh 的 new 段
    #   h' = (1 - z) * n + z * h
    # 结果须与 nn.GRUCell(...) 单步前向一致。
    raise NotImplementedError
