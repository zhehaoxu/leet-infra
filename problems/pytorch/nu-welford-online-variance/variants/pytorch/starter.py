import torch


def solve(chunks, unbiased=True):
    """把按到达顺序发来的一串数据块流式合并成 (mean, var)，只能用 Welford 单遍更新。

    入参：
    - chunks: list/tuple，元素是 float32 或 float64 的 torch.Tensor；每块按 row-major
      展平成标量序列，块内块间都按给定顺序处理。允许出现空块。
    - unbiased: 只接受 bool。True 给样本方差（除以 n-1），False 给总体方差（除以 n）。

    必须维护三个累量 (n, mean, M2)，对每个标量 v 做且只做这一步更新：
        n += 1
        d = v - mean
        mean += d / n
        M2 += d * (v - mean)
    最后 var = M2 / (n - 1)（unbiased）或 M2 / n。全程用 float64 累加。
    **不允许先攒 sum 和 sum of squares 再用 E[x^2] - E[x]^2 收尾**，也不允许把整块数据
    先物化后做两遍统计——那样在数据带大偏置时会灾难性抵消。

    返回 (mean, var) 两个 Python float 组成的 tuple。

    边界与异常：
    - 空流（chunks 为空或所有块都为空，n == 0）返回 (nan, nan)。
    - n == 1：mean 就是那个元素；unbiased=True 时 var 为 nan，unbiased=False 时 var 为 0.0。
    - 流中只要出现过 +inf/-inf/nan，返回 (nan, nan)。
    - chunks 不是 list/tuple、元素不是 Tensor、dtype 非浮点、unbiased 不是 bool，都抛 ValueError。
    """
    # TODO
    raise NotImplementedError
