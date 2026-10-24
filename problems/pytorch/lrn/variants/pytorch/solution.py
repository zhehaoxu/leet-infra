"""参考答案：局部响应归一化 (LRN)。

沿通道维做「邻通道平方和」的除法归一化：对每个 (n, h, w)，把通道展平后取
以 i 为中心、宽 size 的滑动窗口（边界补零），平方和乘 alpha/size 加 k 再取 beta 次幂作分母。
"""

import torch
import torch.nn.functional as F


def solve(x, size, alpha=1e-4, beta=0.75, k=1.0):
    n, c, *rest = x.shape
    x2 = x.reshape(n, c, -1)
    half = size // 2
    sq = x2 * x2
    sqp = F.pad(sq, (0, 0, half, half))                 # 通道维前后各补 half 个零
    sums = torch.zeros_like(sq)
    for j in range(2 * half + 1):                       # size 个偏移累加窗口平方和
        sums = sums + sqp[:, j:j + c, :]
    div = (k + (alpha / size) * sums).pow(beta)
    return (x2 / div).reshape(x.shape)
