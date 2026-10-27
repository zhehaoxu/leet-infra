import torch


def solve(x, p, mask, mode="droppath"):
    """结构化丢弃前向：y = x · mask(广播) / (1 − p)，inverted scaling；mask 显式入参，无随机采样。

    与常规 dropout 的本质差别：丢弃位置不由 RNG 生成，而是显式布尔 / 0-1 张量入参
    （1=保留，0=丢弃），保证 CPU 下完全确定。结构化指按整样本 / 整通道粒度成块丢弃。

    入参：
    - x: float 张量（CPU，float32 / float64）。各 mode 对秩的要求（不符抛 ValueError）：
      * "droppath"：(B, ...) 任意秩 >= 1——整样本丢弃。语义同 timm 的 DropPath；
        出处：Huang et al., "Deep Networks with Stochastic Depth", ECCV 2016
        (arXiv:1603.09382)。
      * "dropout2d"：恰 4 维 (N, C, H, W)——整通道丢弃（同 F.dropout2d 在 training=True
        下的结构语义，但 mask 显式给定而非采样）。
      * "dropout3d"：恰 5 维 (N, C, D, H, W)——整通道（含深度维）丢弃。
    - p: 丢弃概率，须 0 <= p < 1；p >= 1、p < 0 或非有限值抛 ValueError。
      缩放恒为 1/(1−p)（inverted scaling，与 F.dropout training=True 一致）。
    - mask: 显式布尔或 0/1 张量，形状按 mode 二选一（其余形状一律抛 ValueError；
      mask 广播后的形状必须与 x 完全一致，不许广播出额外的 batch 维）：
      * "droppath"：(B,) 或 (B, 1, ..., 1)（1 的个数 = x 维数 − 1）；
      * "dropout2d"：(N, C) 或 (N, C, 1, 1)；
      * "dropout3d"：(N, C) 或 (N, C, 1, 1, 1)。
    - mode: str，"droppath" / "dropout2d" / "dropout3d" 三选一，其余抛 ValueError。

    行为约定：
    - y = (x · mask_广播) / (1 − p)：被丢的样本 / 通道整体严格为 0，存活元素 = x / (1 − p)。
    - p == 0 且 mask 全 1 时严格恒等（torch.equal(y, x)）；p == 0 且 mask 含 0 时
      y == x · mask 逐元素精确相等。
    - mask 为 bool 时先转成 x 的 dtype 再相乘；返回与 x 同形状、同 dtype 的新张量，不修改入参。
    """
    # TODO
    raise NotImplementedError
