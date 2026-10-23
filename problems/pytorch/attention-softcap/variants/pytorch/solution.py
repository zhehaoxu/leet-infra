"""参考答案：注意力 logit soft-capping（Gemma 式）。

在做 softmax 之前，把注意力 logits 用 cap * tanh(logits / cap) 压到 (-cap, cap) 的
有界区间，抑制极端 logit 造成的梯度爆炸与概率塌缩，然后再 softmax。
"""

import torch


def solve(scores, cap):
    capped = cap * torch.tanh(scores / cap)
    return torch.softmax(capped, dim=-1)
