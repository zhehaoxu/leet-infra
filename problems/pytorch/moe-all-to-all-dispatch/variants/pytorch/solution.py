import torch


def solve(hidden_states, topk_indices, topk_weights, num_experts, experts_per_device):
    """MoE EP 场景下 all-to-all 分派 + 加权合并（identity+bias expert）。"""
    T, H = hidden_states.shape
    k = topk_indices.shape[1]
    EPD = int(experts_per_device)
    P = int(num_experts) // EPD
    tpd = (T + P - 1) // P
    send = torch.zeros(P, P, dtype=torch.long)
    for t in range(T):
        src = t // tpd
        if src >= P:
            src = P - 1
        for j in range(k):
            e = int(topk_indices[t, j])
            dst = e // EPD
            send[src, dst] += 1
    weighted_e = (topk_weights.float() * topk_indices.float()).sum(dim=-1, keepdim=True)
    combined = hidden_states.float() + weighted_e
    return send, combined
