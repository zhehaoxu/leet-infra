import torch
import torch.nn.functional as F


def solve(x, expert_ids, routing_weights, gate_w, up_w, down_w,
          shared_gate_w, shared_up_w, shared_down_w):
    gates = torch.einsum("td,tkhd->tkh", x, gate_w[expert_ids])
    values = torch.einsum("td,tkhd->tkh", x, up_w[expert_ids])
    expert_outputs = torch.einsum(
        "tkh,tkdh->tkd", F.silu(gates) * values, down_w[expert_ids]
    )
    routed = (expert_outputs * routing_weights.unsqueeze(-1)).sum(dim=1)
    shared = F.linear(
        F.silu(F.linear(x, shared_gate_w)) * F.linear(x, shared_up_w),
        shared_down_w,
    )
    e, h, d = gate_w.shape
    hs = shared_gate_w.shape[0]
    total_params = 3 * d * (e * h + hs)
    active_params = 3 * d * (expert_ids.shape[1] * h + hs)
    return routed + shared, routed, shared, total_params, active_params
