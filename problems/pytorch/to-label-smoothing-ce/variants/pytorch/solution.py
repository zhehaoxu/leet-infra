import torch
import torch.nn.functional as F


def solve(logits, target, alpha):
    """Label smoothing 交叉熵。"""
    logp = F.log_softmax(logits, dim=-1)
    nll = -logp.gather(1, target.unsqueeze(1)).squeeze(1)
    smooth = -logp.mean(dim=-1)
    return ((1 - alpha) * nll + alpha * smooth).mean()
