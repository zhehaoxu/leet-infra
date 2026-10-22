"""参考答案：自适应平均池化 —— 按 PyTorch 的 floor/ceil 边界公式逐输出位切段求均值。"""

import math

import torch


def _bounds(n_in: int, n_out: int):
    starts = [math.floor(i * n_in / n_out) for i in range(n_out)]
    ends = [math.ceil((i + 1) * n_in / n_out) for i in range(n_out)]
    return list(zip(starts, ends))


def solve(x: torch.Tensor, output_size) -> torch.Tensor:
    if isinstance(output_size, int):
        oh, ow = output_size, output_size
    else:
        oh, ow = output_size
    n, c, h, w = x.shape
    hb, wb = _bounds(h, oh), _bounds(w, ow)
    out = x.new_zeros((n, c, oh, ow))
    for i, (hs, he) in enumerate(hb):
        for j, (ws, we) in enumerate(wb):
            out[:, :, i, j] = x[:, :, hs:he, ws:we].mean(dim=(2, 3))
    return out
