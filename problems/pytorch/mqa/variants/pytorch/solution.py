"""参考答案：多查询注意力 MQA。

把单个共享 KV 头广播到 num_heads 份，逐查询头做缩放点积注意力，最后按头序拼回 hidden。
等价于 KV 头数=1 的 GQA / SDPA(enable_gqa=True)。
"""

import torch


def solve(q, k, v, num_heads):
    b, lq, hid = q.shape
    dh = hid // num_heads
    lk = k.shape[1]
    qh = q.view(b, lq, num_heads, dh).transpose(1, 2)          # (B, nh, Lq, dh)
    kh = k.view(b, lk, 1, dh).transpose(1, 2).expand(b, num_heads, lk, dh)   # 广播单 KV 头
    vh = v.view(b, lk, 1, dh).transpose(1, 2).expand(b, num_heads, lk, dh)
    s = qh @ kh.transpose(-2, -1) / (dh ** 0.5)                # (B, nh, Lq, Lk)
    a = torch.softmax(s, dim=-1)
    return (a @ vh).transpose(1, 2).reshape(b, lq, hid)
