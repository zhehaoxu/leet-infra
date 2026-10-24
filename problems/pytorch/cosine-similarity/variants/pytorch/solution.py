"""参考答案：余弦相似度，对齐 F.cosine_similarity。

cos = (a·b) / max(‖a‖₂, eps) / max(‖b‖₂, eps)，沿 dim 求点积与模长；
分母用 clamp(min=eps)（而非 +eps）以匹配 PyTorch，避免零向量除零。
"""

import torch


def solve(a, b, dim=-1, eps=1e-8):
    dot = (a * b).sum(dim=dim)
    na = a.pow(2).sum(dim=dim).clamp_min(0).sqrt()
    nb = b.pow(2).sum(dim=dim).clamp_min(0).sqrt()
    return dot / na.clamp_min(eps) / nb.clamp_min(eps)
