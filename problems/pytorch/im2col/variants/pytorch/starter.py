import torch


def solve(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None,
          stride=1, padding: int = 0) -> torch.Tensor:
    # TODO: 先用 unfold 把感受野展开成 (N, oh*ow, cin*kh*kw) 的 col 矩阵，
    #   权重 reshape 成 (cin*kh*kw, cout)，做一次 matmul 还原卷积输出
    raise NotImplementedError
