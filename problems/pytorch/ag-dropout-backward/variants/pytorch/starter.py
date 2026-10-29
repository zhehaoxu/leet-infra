import torch


def solve(dY, mask, p, train=True):
    """F.dropout 的反向：把上游梯度按前向的 keep-mask 散射回去，并补上缩放。

    前向（inverted dropout）：y = x · mask / (1 − p)，mask 是 0/1 保留掩码。

    入参：
    - dY: 任意形状的上游梯度；mask: 与 dY 同形、或可广播到 dY 形状（通道级
      dropout2d 就是 (1, C, 1, 1) 这种粗掩码），训练态下形状不可广播时抛 ValueError。
    - p: float，丢弃概率，必须落在 [0, 1)；p=1 或负数抛 ValueError（1−p 作分母）。
    - train: bool。**False 时前向是恒等，反向也必须原样返回 dY，mask 与 p 全部忽略**。

    口径：
    - train=True：dx = dY · mask / (1 − p)，被丢弃的位置梯度为 0，保留位置放大 1/(1−p)。
      训练态下前向与反向**各除一次** (1−p)（PyTorch 实测反向梯度是 mask/(1−p) 而非 mask），
      推理态两处都不除；除此之外不得引入别的缩放。
    - 期望意义下 Σ dx 与 dY 同量级（E[mask]/(1−p) = 1），可作自检。
    - 返回 dtype 与 dY 一致（mask 为 bool/int 也参与运算）。
    """
    # TODO
    raise NotImplementedError
