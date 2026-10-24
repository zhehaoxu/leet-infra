import torch

_EPS = 1e-6


def solve(logits, k):
    """Shazeer OLNN-style CV 辅助损失。"""
    probs = torch.softmax(logits.float(), dim=-1)
    E = probs.shape[1]
    _, sel = torch.topk(probs, k, dim=-1)
    importance = probs.sum(dim=0)
    load = torch.bincount(sel.reshape(-1), minlength=E).float()

    def cv(x):
        return x.std(unbiased=False) / (x.mean() + _EPS)

    return cv(importance) ** 2 + cv(load) ** 2
