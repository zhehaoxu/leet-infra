import torch


def solve(dY, X, W):
    """Y = X @ W.T（即 nn.Linear 的前向）的反向：返回 (dX, dW)，dW 与 W 同形。

    入参：
    - dY: 上游梯度，形状 (..., M, N)，前导可以是任意多个 batch 轴。
    - X:  输入激活，形状 (..., M, K)，前导轴与 dY 完全一致。
    - W:  权重，形状 (N, K) = (out_features, in_features)，必须是二维。

    口径：
    - dX = dY @ W，形状 (..., M, K)。
    - dW = Σ_b dY_b.T @ X_b，形状 (N, K)：**所有前导 batch 轴都要折叠求和**，
      这一点是与 torch.autograd 对拍时最常错的地方（写成单个 (M, N).T 会漏掉求和）。
    - 若 X 只有一行（M=1）或 W 有零尺寸，按同样公式退化处理。
    - 内部 float64 累计，返回两个张量的 dtype 分别与 X、W 相同。

    形状不符（X 末轴 != W 的第 1 轴、dY 末轴 != W 的第 0 轴、前导轴不一致、W 非二维、
    dY/X 少于二维）一律抛 ValueError。
    """
    # TODO
    raise NotImplementedError
