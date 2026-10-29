import torch


def solve(x, gamma, beta, dY, eps=1e-5):
    """LayerNorm（沿末轴）反向：返回 (dX, dGamma, dBeta)。

    前向：mu、var 沿 x 的**最后一个轴**（有偏方差，unbiased=False）计算，
    xhat = (x − mu) / sqrt(var + eps)，y = gamma * xhat + beta。

    入参：
    - x: (..., H) 任意前导轴（可无 batch 轴，即一维 (H,)）。
    - gamma, beta: (H,)，与 x 末轴等长；shape 不符抛 ValueError。
    - dY: 与 x 同形的上游梯度。
    - eps: float，默认 1e-5；允许取 0.0（便于手算示例）。

    口径：
    - dX = ( g − mean(g) − xhat · mean(g · xhat) ) / std，其中 g = dY · gamma，
      mean 都沿末轴。两项减法分别来自「减均值」与「除标准差」两条路径，漏一项必错。
    - dGamma = Σ_前导轴 dY · xhat，dBeta = Σ_前导轴 dY，形状都是 (H,)。
    - beta 不进入任何梯度；对 beta 做扰动结果必须逐位不变。
    - 返回 dtype 与对应入参一致（内部 float64 累计）；dGamma/dBeta 恒为 (H,)，即使 x 只有一维。
    """
    # TODO
    raise NotImplementedError
