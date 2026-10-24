"""参考答案：Transformer 激活显存估算（含梯度检查点）。

按题目规定的简化教学模型清点每层需保存的激活元素数，分别算不启用/启用检查点的字节数。
纯整数计算，用 Python int 避免大数丢精度。
"""

import torch


def solve(b, s, layers, hidden, ffn, heads, dtype_bytes=2):
    # 每层完整保存的激活元素数：QKV(3bs h)+attn out+res/LN(共 2bs h) + FFN 中间(2bs f) + 注意力概率(b a s²)
    e_full = b * s * (6 * hidden + 2 * ffn) + b * heads * s * s
    full_bytes = layers * e_full * dtype_bytes
    # 检查点：常驻每层输入 b*s*hidden，反向峰值再重算一层的完整激活
    checkpoint_bytes = (layers * (b * s * hidden) + e_full) * dtype_bytes
    return full_bytes, checkpoint_bytes
