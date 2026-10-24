"""参考答案：数值稳定的 log-softmax。

log_softmax(x) = x - max(x) - log(sum(exp(x - max(x))))。
先减去沿 dim 的最大值把指数参数压到 <=0，避免 exp 上溢；结果与 F.log_softmax 对齐。
"""

import torch


def solve(x, dim=-1):
    m = x.max(dim=dim, keepdim=True).values
    shifted = x - m
    return shifted - shifted.exp().sum(dim=dim, keepdim=True).log()
