import math


def solve(n, c, k):
    """pass@k 无偏估计：从 n 个采样里已知 c 个通过测试，估计「只采 k 个至少过一个」的概率，
    公式为 1 - C(n-c, k) / C(n, k)。必须用**精确整数**组合数（math.comb）算比值后再转浮点，
    不许用 1 - (1 - c/n)**k 这类有偏近似。

    入参是 Python int，满足 1 <= k <= n 且 0 <= c <= n。返回 Python float ∈ [0, 1]。"""
    # TODO
    raise NotImplementedError
