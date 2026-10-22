"""参考答案：KV Cache 追加 —— 沿序列维把新 token 的 K/V 拼进缓存。"""

import torch


def solve(cache_k: torch.Tensor | None, cache_v: torch.Tensor | None,
          new_k: torch.Tensor, new_v: torch.Tensor, dim: int = -2):
    if cache_k is None or cache_k.numel() == 0:
        return new_k, new_v
    if cache_v is None or cache_v.numel() == 0:
        return torch.cat([cache_k, new_k], dim=dim), new_v
    return torch.cat([cache_k, new_k], dim=dim), torch.cat([cache_v, new_v], dim=dim)
