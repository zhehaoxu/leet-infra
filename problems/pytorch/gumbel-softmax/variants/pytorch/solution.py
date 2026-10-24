"""参考答案：Gumbel-Softmax（重参数化技巧）。

给定 logits 与一组均匀随机数 u∈(0,1)（作为显式输入以保证判题确定性），
先由 u 生成标准 Gumbel 噪声 g = -log(-log(u))，再计算
    y = softmax((logits + g) / tau)
tau→0 时 y 逼近 one-hot（argmax 采样），tau→∞ 时逼近均匀；有限 tau 给出可导的软样本。
"""

import torch
import torch.nn.functional as F


def solve(logits, u, tau=1.0):
    gumbel = -torch.log(-torch.log(u))
    return F.softmax((logits + gumbel) / tau, dim=-1)
