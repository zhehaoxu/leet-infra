import torch


def solve(seq_len: int, num_heads: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    # TODO: bias[h,i,j] = -slope_h * |i-j|，slope_h = 2^(-8*(h+1)/num_heads)；
    #   返回形状 (num_heads, seq_len, seq_len)
    raise NotImplementedError
