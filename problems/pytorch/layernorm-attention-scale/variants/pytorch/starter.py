import torch


def solve(q: torch.Tensor, k: torch.Tensor, weight_q: torch.Tensor, weight_k: torch.Tensor,
          eps: float = 1e-6) -> torch.Tensor:
    # TODO: Q、K 各自沿最后一维 RMSNorm（除 sqrt(mean(x^2)+eps) 再乘各自 weight），
    #   然后 scores = Qn @ Kn^T / sqrt(d)
    raise NotImplementedError
