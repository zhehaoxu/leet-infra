import math

import torch

_MODES = {"eps": "eps", "epsilon": "eps", "x0": "x0", "x_zero": "x0", "v": "v"}


def solve(x_t, alpha_bar, pred, mode):
    if not torch.is_tensor(x_t) or not torch.is_tensor(pred):
        raise ValueError("x_t 与 pred 必须是 torch.Tensor")
    if not isinstance(mode, str) or mode.lower() not in _MODES:
        raise ValueError("mode 必须是 'eps' / 'x0' / 'v'")
    if x_t.shape != pred.shape:
        raise ValueError(f"形状不一致：{tuple(x_t.shape)} vs {tuple(pred.shape)}")

    ab = float(alpha_bar)
    if not (0.0 <= ab <= 1.0):
        raise ValueError("alpha_bar 必须落在 [0, 1] 内")
    kind = _MODES[mode.lower()]
    if kind == "eps" and ab <= 0.0:
        raise ValueError("已知 eps 求 x0 时 ᾱ 不能为 0（除以 α）")
    if kind == "x0" and ab >= 1.0:
        raise ValueError("已知 x0 求 eps 时 ᾱ 不能为 1（除以 σ）")

    x = x_t.to(torch.float64)
    p = pred.to(torch.float64)
    a = math.sqrt(ab)
    s = math.sqrt(1.0 - ab)

    if kind == "eps":
        eps = p
        x0 = (x - s * eps) / a
        v = a * eps - s * x0
    elif kind == "x0":
        x0 = p
        eps = (x - a * x0) / s
        v = a * eps - s * x0
    else:
        v = p
        eps = a * v + s * x
        x0 = a * x - s * v

    return x0.to(torch.float32), eps.to(torch.float32), v.to(torch.float32)
