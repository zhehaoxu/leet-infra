import math

import torch

_EMIN = -6      # 最小规格化指数（2^-6 = 0.015625）
_MAXF = 448.0   # 最大有限值


def _quant_scalar(y: float) -> float:
    if y == 0.0 or math.isnan(y):
        return y
    if math.isinf(y):
        return _MAXF
    s = 1.0 if y > 0 else -1.0
    a = abs(y)
    _, e2 = math.frexp(a)   # a = m * 2^e2, m in [0.5,1) → 实际指数 e = e2-1
    e = e2 - 1
    step = 2.0 ** (e - 3) if e >= _EMIN else 2.0 ** (_EMIN - 3)  # 次正规固定 2^-9
    n = float(round(a / step))   # Python round = RTNE（平局取偶）
    r = n * step
    if r > _MAXF:
        r = _MAXF                # 饱和
    return s * r


def solve(x: torch.Tensor) -> torch.Tensor:
    flat = x.reshape(-1)
    out = torch.empty_like(flat)
    for i in range(flat.numel()):
        out[i] = _quant_scalar(float(flat[i]))
    return out.reshape(x.shape)
