import torch


def solve(logits, target, label_smoothing=0.0, ignore_index=-100, reduction='mean'):
    """交叉熵反向：给定 logits 与 target，返回损失对 logits 的梯度（与 logits 同形）。

    入参：
    - logits: (N, C) 二维张量，未经 softmax 的分数。
    - target: (N,) 一维 int64 类别下标；等于 ignore_index 的位置视为被忽略。
    - label_smoothing: float，∈ [0, 1)，越界抛 ValueError。
    - ignore_index: int，默认 -100。
    - reduction: 'mean' 或 'sum'，其它值抛 ValueError。

    口径（与 F.cross_entropy 一致）：
    - 单个有效行的梯度 dL/dz_k = p_k − (1 − ε)·onehot_y[k] − ε / C，其中 p = softmax(logits)。
      ε=0 时退化为常见的 p − onehot。
    - 被 ignore 的行梯度整行为 0，且**不计入 mean 的分母**（分母是有效行数，不是 N）。
    - 'sum' 不做任何除法。
    - 内部用 float64 累计，返回值 dtype 与 logits 相同。

    校验：logits 非二维、target 非一维或长度与 N 不符时抛 ValueError。
    """
    # TODO
    raise NotImplementedError
