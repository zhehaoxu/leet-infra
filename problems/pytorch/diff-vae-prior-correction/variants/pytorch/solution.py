import math

import torch


def _stat(value, channels, name):
    if not torch.is_tensor(value):
        try:
            value = torch.as_tensor(value, dtype=torch.float64)
        except Exception:
            raise ValueError(f"{name} 必须是张量或数字序列")
    t = value.to(torch.float64)
    if not torch.isfinite(t).all():
        raise ValueError(f"{name} 里不能有 nan / inf")
    if t.numel() == 1:
        return float(t.reshape(-1)[0])
    if t.dim() != 1:
        raise ValueError(f"{name} 必须是一维（每通道一个值）")
    if t.numel() != channels:
        raise ValueError(f"{name} 长度 {t.numel()} 与通道数 {channels} 不一致")
    return t


def _bcast(v, ndim, channels):
    if isinstance(v, float):
        return v
    return v.reshape((1, channels) + (1,) * (ndim - 2))


def solve(latents, scaling_factor=1.0, shift_factor=0.0, channel_mean=None, channel_std=None, inverse=False):
    if not torch.is_tensor(latents):
        raise ValueError("latents 必须是 torch.Tensor")
    if latents.dim() < 2:
        raise ValueError("latents 至少要有两维（batch × 通道）")

    scale = float(scaling_factor)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("scaling_factor 必须是正的有限实数")
    shift = float(shift_factor)
    if not math.isfinite(shift):
        raise ValueError("shift_factor 必须是有限实数")

    if (channel_mean is None) != (channel_std is None):
        raise ValueError("channel_mean 与 channel_std 必须同时给或同时不给")

    channels = latents.shape[1]
    mean = std = None
    if channel_mean is not None:
        mean = _stat(channel_mean, channels, "channel_mean")
        std = _stat(channel_std, channels, "channel_std")
        if isinstance(std, float):
            if std <= 0.0:
                raise ValueError("channel_std 必须为正")
        elif bool((std <= 0.0).any()):
            raise ValueError("channel_std 必须全为正")

    x = latents.to(torch.float64)
    ndim = latents.dim()

    if inverse:
        raw = (x - shift) * scale
        if std is not None:
            raw = (raw - _bcast(mean, ndim, channels)) / _bcast(std, ndim, channels)
        return raw.to(torch.float32)

    raw = x
    if std is not None:
        raw = raw * _bcast(std, ndim, channels) + _bcast(mean, ndim, channels)
    return (raw / scale + shift).to(torch.float32)
