import torch


def solve(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None,
          stride=1, padding: int = 0) -> torch.Tensor:
    # TODO: 三重循环：输出位置 (i,j) × 输出通道 o，
    #   out[n,o,i,j] = Σ_c Σ_u Σ_v x_pad[n,c,i*s+u,j*s+v] * weight[o,c,u,v]
    raise NotImplementedError
