import torch


def solve(W0, theta, r, alpha, proj_seed, layer_id=0):
    """UniLoRA 合并：从共享向量 theta 中按 proj_seed+layer_id 固定索引取出 A/B，
    计算 W' = W0 + (alpha/r) * B @ A。theta: (D,) W0:(d_out,d_in)。返回 W':(d_out,d_in)。"""
    # TODO
    raise NotImplementedError
