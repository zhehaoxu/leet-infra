import math

import torch

_WINDOWS = ("hann", "hamming", "rect")


def _window(name, n):
    """对称窗（periodic=False 口径），n == 1 时退化为 [1]。"""
    if n == 1:
        return torch.ones(1, dtype=torch.float64)
    pos = torch.arange(n, dtype=torch.float64)
    if name == "hann":
        return 0.5 - 0.5 * torch.cos(2.0 * math.pi * pos / (n - 1))
    if name == "hamming":
        return 0.54 - 0.46 * torch.cos(2.0 * math.pi * pos / (n - 1))
    return torch.ones(n, dtype=torch.float64)


def solve(wave, preemph, frame_len, hop_len, window="hann"):
    if not isinstance(wave, torch.Tensor) or wave.dim() != 1:
        raise ValueError("wave 必须是一维张量")
    if window not in _WINDOWS:
        raise ValueError("window 必须是 hann / hamming / rect 之一")
    if frame_len < 1 or hop_len < 1:
        raise ValueError("frame_len 与 hop_len 必须为正")
    if not (0.0 <= float(preemph) < 1.0):
        raise ValueError("preemph 必须落在 [0, 1)")

    x = wave.to(torch.float64)
    total = int(x.numel())
    if total < frame_len:
        raise ValueError("波形长度不能短于 frame_len")

    # 预加重：首样本原样保留，其后做一阶差分
    y = x.clone()
    if total > 1:
        y[1:] = x[1:] - float(preemph) * x[:-1]

    num_frames = 1 + (total - frame_len) // hop_len
    starts = torch.arange(num_frames, dtype=torch.int64) * hop_len
    idx = starts[:, None] + torch.arange(frame_len, dtype=torch.int64)[None, :]
    frames = y[idx] * _window(window, frame_len)[None, :]

    return y.to(torch.float32), frames.to(torch.float32)
