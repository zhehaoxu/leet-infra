import math

import torch

_SCHEMES = ("logit_normal", "mode", "uniform")


def solve(scheme, samples, logit_mean=None, logit_std=None, mode_scale=None):
    if not isinstance(scheme, str) or scheme not in _SCHEMES:
        raise ValueError("scheme 必须是 logit_normal / mode / uniform 之一")
    if not torch.is_tensor(samples):
        raise ValueError("samples 必须是 torch 张量")
    if samples.ndim != 1:
        raise ValueError("samples 必须是一维张量")

    s = samples.to(torch.float64)
    if not bool(torch.isfinite(s).all()):
        raise ValueError("samples 不能含 nan / inf")

    if scheme == "logit_normal":
        if logit_mean is None or logit_std is None:
            raise ValueError("logit_normal 需要同时给出 logit_mean 与 logit_std")
        mean = float(logit_mean)
        std = float(logit_std)
        if not (math.isfinite(mean) and math.isfinite(std)) or std < 0:
            raise ValueError("logit_mean / logit_std 必须是有限实数，且 logit_std >= 0")
        out = torch.sigmoid(mean + std * s)
    else:
        if bool(((s < 0) | (s > 1)).any()):
            raise ValueError(f"{scheme} 的 samples 必须落在 [0, 1]")
        if scheme == "mode":
            if mode_scale is None:
                raise ValueError("mode 需要给出 mode_scale")
            scale = float(mode_scale)
            if not math.isfinite(scale) or scale < 0:
                raise ValueError("mode_scale 必须是有限非负实数")
            u = s
            out = 1 - u - scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
        else:
            out = s

    return out.to(torch.float32)
