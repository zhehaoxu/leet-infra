import torch


def solve() -> list[float]:
    # TODO: w=[2.0] 起步，SGD(lr=0.1, momentum=0.9) 跑 20 步，
    # 每步 zero_grad -> loss=(w**2).sum() -> 记录 -> backward -> step
    raise NotImplementedError
