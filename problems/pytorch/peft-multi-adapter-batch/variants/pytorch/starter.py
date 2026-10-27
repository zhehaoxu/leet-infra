import torch


def solve(W0, A, B, scales, ids, x):
    """多 adapter 批量路由前向：y_i = W0 x_i + s_{ids[i]} * B_{ids[i]} A_{ids[i]} x_i。
    W0:(d_out,d_in)；A:(K,r,d_in) 下投影；B:(K,d_out,r) 上投影；scales:(K,)；ids:(N,)；x:(N,d_in)。
    返回 y:(N,d_out) float32。禁止现成 PEFT/vLLM 库。"""
    # TODO
    raise NotImplementedError
