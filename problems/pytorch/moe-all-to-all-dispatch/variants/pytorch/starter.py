import torch


def solve(hidden_states, topk_indices, topk_weights, num_experts, experts_per_device):
    """EP all-to-all：返回 (send_matrix (P,P) long, combined (T,H) float)。
    src_device(t) = t // ((T+P-1)//P); dst_device(e) = e // EPD。
    combined[t] = hidden[t] + Σ_j weight[t,j] * topk_indices[t,j]。
    """
    # TODO
    raise NotImplementedError
