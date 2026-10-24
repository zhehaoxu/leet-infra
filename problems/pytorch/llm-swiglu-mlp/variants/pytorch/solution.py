import torch
import torch.nn.functional as F


def solve(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    gate = F.silu(F.linear(x, gate_weight))
    up = F.linear(x, up_weight)
    return F.linear(gate * up, down_weight)
