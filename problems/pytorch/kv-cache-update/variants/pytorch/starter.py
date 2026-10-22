import torch


def solve(cache_k: torch.Tensor | None, cache_v: torch.Tensor | None,
          new_k: torch.Tensor, new_v: torch.Tensor, dim: int = -2):
    # TODO: 把新 token 的 K/V 沿序列维（dim=-2）追加到缓存后返回 (k, v)；
    #   缓存为 None 或空张量时直接返回新值
    raise NotImplementedError
