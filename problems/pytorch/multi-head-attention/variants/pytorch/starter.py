import torch


def solve(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int,
          mask: torch.Tensor | None = None) -> torch.Tensor:
    # TODO: view(B,L,H,dh)+transpose(1,2) 拆头；逐头缩放点积注意力；
    #   transpose(1,2)+reshape 合头。mask 形状 (...,Lq,Lk) 或 None
    raise NotImplementedError
