import torch


def solve(weight, activations, grad_samples, sparsity, scale=1.0,
          gradient_inv=False, prune_n=0, prune_m=0):
    """GBLM：|W|·√s_row ⊕ 梯度项 的融合显著性逐行掩码（待实现）。"""
    raise NotImplementedError
