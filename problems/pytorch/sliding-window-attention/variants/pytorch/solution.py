"""参考答案：因果 + 滑动窗口注意力。

位置 i 的 token 只能回看区间 [i-window+1, i] 内的 key/value：既是因果的
（j <= i），又有窗口上限（i - j < window）。window >= 序列长度时退化为纯因果注意力。
对角线恒可见（i 总能看到自己），保证每行至少有一个非 -inf 的 logits。
"""

import math

import torch


def solve(q, k, v, window):
    b, l, d = q.shape
    scale = 1.0 / math.sqrt(d)
    s = torch.matmul(q, k.transpose(-2, -1)) * scale                         # (b,l,l)
    idx = torch.arange(l, device=q.device)
    causal = idx.view(1, l, 1) >= idx.view(1, 1, l)                          # j <= i
    within = (idx.view(1, l, 1) - idx.view(1, 1, l)) < window                # i - j < window
    mask = causal & within                                                   # True = 可见
    s = s.masked_fill(~mask, float("-inf"))
    a = torch.softmax(s, dim=-1)
    return torch.matmul(a, v)
