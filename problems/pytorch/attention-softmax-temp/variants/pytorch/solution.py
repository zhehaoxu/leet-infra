"""参考答案：温度缩放 softmax + top-p (nucleus) 截断重归一化。"""

import torch


def solve(logits: torch.Tensor, temperature: float = 1.0, top_p: float | None = None) -> torch.Tensor:
    scaled = logits / temperature
    shifted = scaled - scaled.max(dim=-1, keepdim=True).values
    exp = torch.exp(shifted)
    probs = exp / exp.sum(dim=-1, keepdim=True)
    if top_p is None or top_p >= 1.0:
        return probs
    sorted_probs, idx = probs.sort(dim=-1, descending=True)
    # nucleus：保留累计概率「不含自己」仍 < top_p 的项（保证至少留 1 项）
    cum = sorted_probs.cumsum(dim=-1)
    remove = (cum - sorted_probs) >= top_p
    kept = sorted_probs.masked_fill(remove, 0.0)
    kept = kept / kept.sum(dim=-1, keepdim=True)
    return torch.zeros_like(probs).scatter_(-1, idx, kept)
