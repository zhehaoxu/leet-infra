import math
import torch

NF4 = [-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
       0.0795, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7226, 1.0]


def solve(codes, absmax, x, A, B, alpha, r):
    """QLoRA NF4 反量化前向 + double-quant 字节账。
    codes:(d_out,d_in) int64 且 d_in%64==0；absmax:(d_out,d_in//64)；A:(r,d_in)、B:(d_out,r)。
    W[i,j]=NF4[codes[i,j]]*absmax[i,j//64]，y = x@W^T + (alpha/r)*(x@A^T)@B^T；
    nbytes = n/2 + n/64 + ceil(n/64/256)*4（n=d_out*d_in）。
    返回 (y:(N,d_out) float32, nbytes:int)。NF4 码本见题面。禁止 bitsandbytes。"""
    table = torch.tensor(NF4, dtype=torch.float64)
    codes = codes.to(torch.float64)
    scales = absmax.to(torch.float64).repeat_interleave(64, dim=1)   # (d_out, d_in)
    W = table[codes.long()] * scales
    x64 = x.to(torch.float64)
    base = x64 @ W.T
    lora = (x64 @ A.to(torch.float64).T) @ B.to(torch.float64).T
    y = base + (float(alpha) / float(r)) * lora
    n = codes.shape[0] * codes.shape[1]
    n_blocks = n // 64
    nbytes = n // 2 + n_blocks + math.ceil(n_blocks / 256) * 4
    return y.to(torch.float32), int(nbytes)
