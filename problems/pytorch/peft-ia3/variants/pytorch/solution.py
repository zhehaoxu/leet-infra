"""参考答案：IA3 —— 线性输出后逐输出维乘性重标定。"""

import torch


def solve(x, W, lam):
    h = x @ W.T
    return h * lam
