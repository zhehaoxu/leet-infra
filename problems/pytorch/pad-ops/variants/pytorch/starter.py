import torch


def solve(x: torch.Tensor, padding: tuple[int, ...], mode: str = "constant", value: float = 0.0) -> torch.Tensor:
    # TODO: 常量 padding。padding 成对给出且从**最后一个维**向前排：
    #   (last_left, last_right, 2ndlast_left, 2ndlast_right, ...)；
    #   新位置填 value。返回补边后的张量。
    raise NotImplementedError
