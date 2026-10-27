import torch


def solve(scales, bits, block_size):
    """double quantization 的二级量化：按 block_size 连续分组，对称定点量化一级 scale。
    scales:(B,) float32；bits 在 [2,8]；block_size>0。
    每组 s2=max|s|/(2^(bits-1)-1)（全零组取 1），
    code=clamp(round(s/s2),-qmax,qmax)，deq=code*s2。
    返回 (codes:(B,) int64, group_scale:(ceil(B/block_size),) float32, deq:(B,) float32)。
    bits/block_size 非法抛 ValueError。"""
    # TODO
    raise NotImplementedError
