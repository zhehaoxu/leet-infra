"""参考答案：分组查询注意力 GQA / MQA。

q 有 num_heads 个头，k/v 只有 num_kv_heads 个头，每 (num_heads // num_kv_heads)
个连续的 query 头共享同一组 kv 头（repeat_interleave 的「成组」语义，非循环取模）。
num_kv_heads==num_heads 退化为多头注意力 MHA，==1 退化为多查询注意力 MQA。
"""

import torch


def solve(q, k, v, num_heads, num_kv_heads):
    b, lq, hd = q.shape
    dh = hd // num_heads
    lk = k.shape[1]
    qh = q.view(b, lq, num_heads, dh).transpose(1, 2)        # (b,H,lq,dh)
    kh = k.view(b, lk, num_kv_heads, dh).transpose(1, 2)     # (b,KV,lk,dh)
    vh = v.view(b, lk, num_kv_heads, dh).transpose(1, 2)
    rep = num_heads // num_kv_heads
    if rep > 1:
        kh = kh.repeat_interleave(rep, dim=1)                # (b,H,lk,dh)
        vh = vh.repeat_interleave(rep, dim=1)
    s = torch.matmul(qh, kh.transpose(-2, -1)) / (dh ** 0.5)
    a = torch.softmax(s, dim=-1)
    o = torch.matmul(a, vh)                                  # (b,H,lq,dh)
    return o.transpose(1, 2).reshape(b, lq, hd)
