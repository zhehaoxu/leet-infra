import torch


def solve(x, gamma, dY, eps=1e-6):
    """RMSNorm（沿末轴）反向：返回 (dX, dGamma)。

    前向：rms = sqrt(mean(x², 末轴) + eps)，xhat = x / rms，y = gamma * xhat。
    与 LayerNorm 的关键差别：**不减均值**，x 的均值分量不参与归一化，因此 x 本身
    （而非 x − mu）是那条唯一的耦合方向。

    入参：
    - x: (..., H)，任意前导轴。
    - gamma: (H,)；dY: 与 x 同形；eps: float，允许 0.0，负值抛 ValueError。
    - gamma 长度与末轴不符、dY 形状不符一律抛 ValueError。

    口径：
    - g = dY · gamma，dX = ( g − xhat · mean(g · xhat) ) / rms（mean 沿末轴）。
      与 LayerNorm 的公式相比少了 mean(g) 那一项；把它照抄过来会在 gamma 全 1、
      dY 常数时给出 0，而正确的 RMSNorm 梯度此时应非 0。
    - dGamma = Σ_前导轴 dY · xhat，形状 (H,)。
    - 返回 dtype 与入参一致（内部 float64 累计）。
    """
    # TODO
    raise NotImplementedError
