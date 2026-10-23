"""参考答案：FlashAttention 风格的分块在线 softmax。

不一次性算出完整注意力矩阵，而是把 query 分块，对每块沿 key 维分块遍历，
用 running max / running sum 做在线（numerically-stable）softmax 归约，
逐块 rescale 累加。最终结果与标准 softmax(qk^T/sqrt(d)) v 数值等价。
"""

import math

import torch


def solve(q, k, v, block=2):
    b, lq, d = q.shape
    lk = k.shape[1]
    scale = 1.0 / math.sqrt(d)
    out = torch.zeros(b, lq, d, dtype=q.dtype, device=q.device)
    for i0 in range(0, lq, block):
        i1 = min(i0 + block, lq)
        qb = q[:, i0:i1, :]                                                  # (b,m,d)
        m = i1 - i0
        m_i = torch.full((b, m, 1), float("-inf"), dtype=q.dtype, device=q.device)
        l_i = torch.zeros(b, m, 1, dtype=q.dtype, device=q.device)
        acc = torch.zeros(b, m, d, dtype=q.dtype, device=q.device)
        for j0 in range(0, lk, block):
            j1 = min(j0 + block, lk)
            kb = k[:, j0:j1, :]
            vb = v[:, j0:j1, :]
            s = torch.matmul(qb, kb.transpose(-2, -1)) * scale               # (b,m,n)
            m_new = torch.maximum(m_i, s.max(dim=-1, keepdim=True).values)
            p = torch.exp(s - m_new)
            alpha = torch.exp(m_i - m_new)                                   # 首块 m_i=-inf → alpha=0
            l_i = l_i * alpha + p.sum(dim=-1, keepdim=True)
            acc = acc * alpha + torch.matmul(p, vb)
            m_i = m_new
        out[:, i0:i1, :] = acc / l_i
    return out
