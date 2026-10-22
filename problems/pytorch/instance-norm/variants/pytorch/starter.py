import torch


def solve(x: torch.Tensor, weight: torch.Tensor | None = None, bias: torch.Tensor | None = None, eps: float = 1e-5) -> torch.Tensor:
    # TODO: 逐样本逐通道：对 dim>=2 的空间维求均值/有偏方差归一化；
    #   weight/bias 可能为 None，非 None 时按通道广播
    raise NotImplementedError
