import torch


def solve(logits, k):
    """Top-k MoE gating，与 HF MixtralTopKRouter.forward 一致。

    返回 (scores (T,k) float, indices (T,k) long)。
    """
    probs = torch.softmax(logits.float(), dim=-1)
    tv, ti = torch.topk(probs, k, dim=-1)
    scores = tv / tv.sum(dim=-1, keepdim=True)
    return scores, ti
