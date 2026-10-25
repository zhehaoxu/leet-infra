"""参考答案：Codex 论文的 pass@k 无偏估计量，用精确整数组合数计算。"""

import math


def solve(n, c, k):
    num = math.comb(n - c, k)          # 全错的 k-子集数（n-c < k 时自动为 0）
    den = math.comb(n, k)              # 全部 k-子集数
    return 1.0 - num / den
