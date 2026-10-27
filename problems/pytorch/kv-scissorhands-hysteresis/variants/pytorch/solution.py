import torch


def solve(scores, tau, patience):
    """ScissorHands 滞回驱逐（逐 token 连续低分计数 + 首次达标即驱逐）。"""
    patience = int(patience)
    if patience < 1:
        raise ValueError("patience 必须 >= 1")
    s_steps, length = scores.shape
    tau = float(tau)

    streak = torch.zeros(length, dtype=torch.long)
    evict = torch.full((length,), -1, dtype=torch.long)
    alive = torch.ones(length, dtype=torch.bool)
    keep = torch.ones(s_steps, length, dtype=torch.bool)
    for s in range(s_steps):
        low = scores[s] < tau
        streak = torch.where(low, streak + 1, torch.zeros_like(streak))
        hit = (streak >= patience) & alive
        evict = torch.where(hit, torch.full_like(evict, s), evict)
        alive = alive & ~hit
        keep[s] = alive
    return evict, keep
