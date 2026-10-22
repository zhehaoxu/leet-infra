import torch


def solve(x: torch.Tensor, p: float = 0.5, training: bool = True, seed: int = 1234) -> torch.Tensor:
    # TODO: training=False 或 p=0 时恒等；否则用
    #   torch.Generator(device=x.device).manual_seed(seed) 生成
    #   mask = rand(shape, generator=g) >= p，输出 x*mask/(1-p)（inverted scaling）
    raise NotImplementedError
