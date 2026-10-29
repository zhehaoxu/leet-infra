import torch


def solve(x, dY, kind, negative_slope=0.01):
    """逐元素激活的反向：dx = dY · φ'(x)，按 kind 选解析式（前向留在 operators 合集）。

    入参：
    - x: 任意形状；dY: 与 x 同形；kind: 字符串，取以下之一，其它值抛 ValueError：
      'relu'、'leaky_relu'、'elu'、'sigmoid'、'tanh'、'gelu'、'silu'、'softplus'。
    - negative_slope: 只被 'leaky_relu' 使用（其余 kind 忽略它，但不得报错）。

    解析梯度（与 F.* 的 autograd 逐位一致）：
    - relu: 1 若 x > 0，否则 0。**x == 0 处取 0**（PyTorch 的次梯度约定，取 0.5 必错）。
    - leaky_relu: 1 若 x > 0，否则 negative_slope。
    - elu: 1 若 x > 0，否则 exp(x)。
    - sigmoid: s(1 − s)；tanh: 1 − t²；softplus: sigmoid(x)。
    - gelu: 精确型 Φ(x) + x·φ(x)，Φ(x) = 0.5(1 + erf(x/√2))、φ 是标准正态密度。
    - silu: σ(x) + x·σ(x)(1 − σ(x))。

    要求：内部 float64 计算、返回 dtype 与 x 一致；x 与 dY 形状不符抛 ValueError；
    导数在 x=0 的取值：relu/leaky_relu/elu/silu/softplus 分别为 0 / negative_slope / 1 / 0.5 / 0.5。
    """
    # TODO
    raise NotImplementedError
