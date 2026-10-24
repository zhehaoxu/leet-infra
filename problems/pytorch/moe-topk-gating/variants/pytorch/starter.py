import torch


def solve(logits, k):
    """Top-k MoE gating: probs=softmax(logits,-1); (v,i)=topk(probs,k,-1); scores=v/v.sum(-1,keepdim).
    返回 (scores (T,k) float, indices (T,k) long)。
    """
    # TODO
    raise NotImplementedError
