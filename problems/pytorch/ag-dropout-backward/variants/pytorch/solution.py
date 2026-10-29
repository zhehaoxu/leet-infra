"""参考答案：train 态是「掩码 + 一次 1/(1−p) 缩放」，eval 态是恒等。

inverted dropout 的缩放出现在前向 y = x·mask/(1−p)，链式法则对 x 求导又给出同一个
1/(1−p)，所以实测 dx = dY·mask/(1−p)（PyTorch 反向值域是 {0, 2} at p=0.5）；
若是旧式「前向不缩放、推理时缩权重」的写法，反向才只乘 mask，两种约定混用会多除一次。
mask 允许是低维可广播张量，因此用 reshape 而不是逐元素索引；eval 态必须整条跳过，
否则部署时会凭空把梯度放大 1/(1−p)。
"""

import torch


def solve(dY, mask, p, train=True):
    if p is None or not (0.0 <= float(p) < 1.0):
        raise ValueError("p 必须落在 [0, 1) 内")
    if not isinstance(train, bool):
        raise ValueError("train 必须是 bool")
    if not train:
        return dY.clone() if isinstance(dY, torch.Tensor) else dY

    if tuple(mask.shape) != tuple(dY.shape):
        try:
            m = mask.expand_as(dY)
        except RuntimeError:
            raise ValueError("mask 必须与 dY 同形或可广播到 dY 形状")
    else:
        m = mask

    keep = m.to(dY.dtype)
    return dY * keep / (1.0 - float(p))
