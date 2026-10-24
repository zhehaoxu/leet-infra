"""参考答案：非对称量化 fake-quant —— per-tensor (max-min)/255 + zero_point，round→clamp→dequant。"""

import torch


def solve(x: torch.Tensor) -> torch.Tensor:
    xmin = x.min()
    xmax = x.max()
    scale = (xmax - xmin) / 255.0
    if float(scale) == 0.0:  # 常数张量，退化
        return x.clone()
    # zero_point 夹进表示范围 [0,255]（全正/全负张量时未夹会越界）
    zp = torch.clamp(torch.round(-xmin / scale), 0, 255)
    q = torch.clamp(torch.round(x / scale) + zp, 0, 255)
    return (q - zp) * scale
