import torch


def solve(b, s, layers, hidden, ffn, heads, dtype_bytes=2):
    # TODO: 估算 Transformer 激活显存，返回 (full_bytes, checkpoint_bytes)，均为整数。
    # 每层完整保存激活（元素数）： E_full = b*s*(6*hidden + 2*ffn) + b*heads*s*s
    # 不启用检查点： full_bytes = layers * E_full * dtype_bytes
    # 启用检查点：   checkpoint_bytes = (layers * b*s*hidden + E_full) * dtype_bytes
    #   （常驻每层输入 b*s*hidden，反向峰值额外重算 1 层的完整激活）
    raise NotImplementedError
