"""参考答案：对称量化 fake-quant —— per-tensor absmax/127，round→clamp→dequant 往返。"""

import torch


def solve(x: torch.Tensor) -> torch.Tensor:
    amax = x.abs().max()
    if float(amax) == 0.0:
        return x.clone()
    scale = amax / 127.0
    q = torch.clamp(torch.round(x / scale), -127, 127)
    return q * scale
