import torch


def solve(codes, absmax, x, A, B, alpha, r):
    """QLoRA NF4 反量化前向 + double-quant 字节账。
    codes:(d_out,d_in) int64 且 d_in%64==0；absmax:(d_out,d_in//64)；A:(r,d_in)、B:(d_out,r)。
    W[i,j]=NF4[codes[i,j]]*absmax[i,j//64]，y = x@W^T + (alpha/r)*(x@A^T)@B^T；
    nbytes = n/2 + n/64 + ceil(n/64/256)*4（n=d_out*d_in）。
    返回 (y:(N,d_out) float32, nbytes:int)。NF4 码本见题面。禁止 bitsandbytes。"""
    # TODO
    raise NotImplementedError
