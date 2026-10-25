import torch


def solve(initial_sparsity, final_sparsity, current_step, first_step, final_step):
    """三次渐稀疏调度(Zhu-Gupta 1710.01878)：p_t=final+(initial-final)*(1-(t-t0)/(t1-t0))^3。
    cur<=first 返回 initial；cur>=final_step 返回 final。返回 0 维 float32 标量。禁止现成调度库。"""
    # TODO
    raise NotImplementedError
