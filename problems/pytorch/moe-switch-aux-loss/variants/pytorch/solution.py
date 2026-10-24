import torch


def solve(logits, k):
    """Switch 负载均衡辅助损失（HF Mixtral `load_balancing_loss_func` 语义）。

    loss = E * Σ_e f_e * P_e,  f_e = count_e/T, P_e = probs.mean(0)
    """
    probs = torch.softmax(logits.float(), dim=-1)
    T, E = probs.shape
    _, sel = torch.topk(probs, k, dim=-1)
    f = torch.bincount(sel.reshape(-1), minlength=E).float() / T
    P = probs.mean(dim=0)
    return E * (f * P).sum()
