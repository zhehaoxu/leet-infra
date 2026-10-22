import torch


def solve(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
):
    # TODO: training=True 用 batch 均值/有偏方差归一化，并按
    #   new = (1-momentum)*old + momentum*batch 更新 running stats；
    #   training=False 用 running stats。返回 (y, running_mean_new, running_var_new)
    raise NotImplementedError
