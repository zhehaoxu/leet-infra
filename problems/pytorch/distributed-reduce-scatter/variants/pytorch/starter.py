import torch


def solve(inputs: list[torch.Tensor]) -> list[torch.Tensor]:
    # TODO: 逐元素求和得 T（长度 L），均分 W 段，rank i 取 T[i*chunk:(i+1)*chunk]，chunk=L//W
    raise NotImplementedError
