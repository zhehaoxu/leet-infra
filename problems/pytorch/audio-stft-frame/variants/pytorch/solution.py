import math

import torch
import torch.nn.functional as F


def solve(wave, n_fft, hop_len, center):
    if not isinstance(wave, torch.Tensor) or wave.dim() != 1:
        raise ValueError("wave 必须是一维张量")
    if n_fft < 2 or n_fft % 2 != 0:
        raise ValueError("n_fft 必须是 >= 2 的偶数")
    if hop_len < 1 or hop_len > n_fft:
        raise ValueError("hop_len 必须落在 [1, n_fft]")

    x = wave.to(torch.float64)
    total = int(x.numel())

    if center:
        pad = n_fft // 2
        if total <= pad:
            raise ValueError("center=True 时 T 必须大于 n_fft//2")
        # 反射填充：不重复边界点，等价于 torch.stft(center=True, pad_mode='reflect')
        x = F.pad(x.view(1, 1, -1), (pad, pad), mode="reflect").view(-1)
        num_frames = 1 + total // hop_len
    else:
        if total < n_fft:
            raise ValueError("center=False 时 T 不能小于 n_fft")
        num_frames = 1 + (total - n_fft) // hop_len

    pos = torch.arange(n_fft, dtype=torch.float64)
    win = 0.5 - 0.5 * torch.cos(2.0 * math.pi * pos / n_fft)

    starts = torch.arange(num_frames, dtype=torch.int64) * hop_len
    idx = starts[:, None] + torch.arange(n_fft, dtype=torch.int64)[None, :]
    frames = x[idx] * win[None, :]

    spec = torch.fft.rfft(frames, n=n_fft, dim=1)
    return spec.abs().to(torch.float32)
