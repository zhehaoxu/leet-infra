import torch


def solve(feature, points, offsets, weights=None):
    """可变形（deformable）双线性采样与可选加权聚合。

    入参：
    - feature: (C, H, W) 的 float32/float64 张量，C、H、W 都必须 >= 1。像素
      坐标约定：x 是列下标（0..W-1），y 是行下标（0..H-1）。
    - points: (N, K, 2) 的同 dtype 张量，最后一维是 (x, y) 像素坐标；可以是
      小数，也可以落在 [0, W-1] x [0, H-1] 之外。N 可以为 0，K 必须 >= 1。
    - offsets: (N, K, 2) 的同 dtype 张量，形状必须与 points 完全一致，最后一维
      是 (dx, dy)；实际采样位置为 (x + dx, y + dy)。
    - weights: 默认 None。给张量时必须是 (N, K)、同 dtype；此时返回按采样点
      加权求和后的 (N, C)。

    采样口径（双线性 + 逐角零填充）：
    - 对采样位置 (x, y) 令 x0 = floor(x)、y0 = floor(y)、fx = x - x0、
      fy = y - y0（floor 是向下取整，负数也一样，例如 floor(-0.5) = -1）；
    - 四个角是 (x0, y0)、(x0+1, y0)、(x0, y0+1)、(x0+1, y0+1)，权重分别为
      (1-fx)(1-fy)、fx(1-fy)、(1-fx)fy、fx*fy，取值为 feature[:, y角, x角]；
    - **落在 [0, W-1] x [0, H-1] 之外的角一律按 0 参与**（zero padding），不做
      夹取/复制边缘、也不做坐标截断；因此采样位置越界只留下仍然在界内的角的
      贡献，全部角越界时结果为 0；
    - 角坐标恰为 H-1 / W-1 时，多出来的一侧权重为 0，不影响结果。

    返回：
    - weights 为 None：返回 (N, K, C) 张量，第 [n, k, c] 项是第 n 组第 k 个
      采样位置的 c 通道双线性取值；
    - weights 给定：返回 (N, C) 张量，out[n, c] = sum_k weights[n, k] *
      sampled[n, k, c]，**不做归一化**（不除以权重之和），所以权重不必和为 1。
    - 输出 dtype 跟随 feature；内部几何一律在 float64 中计算再转回。

    非法输入一律抛 ValueError：feature 不是张量或不是三维、C/H/W < 1、dtype
    不是 float32/float64；points/offsets 不是张量、不是 (N,K,2)、两者形状不
    一致、dtype 与 feature 不同、K < 1；weights 不是张量或形状不是 (N,K) 或
    dtype 不同；points/offsets/weights 含 NaN 或 Inf。不修改任何输入。
    """
    # TODO
    raise NotImplementedError
