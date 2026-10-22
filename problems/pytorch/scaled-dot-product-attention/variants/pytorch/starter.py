import torch


def solve(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
          mask: torch.Tensor | None = None) -> torch.Tensor:
    # TODO: scores = QK^T/sqrt(d_k)；bool mask 中 True=允许注意力、False=屏蔽(置 -inf)，与
    #   F.scaled_dot_product_attention 一致；整行被屏蔽时输出 0；行方向数值稳定 softmax；
    #   再乘 V。支持 (..., L, d) 批量
    raise NotImplementedError
