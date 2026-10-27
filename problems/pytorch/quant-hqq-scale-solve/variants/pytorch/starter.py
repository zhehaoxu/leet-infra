import torch


def solve(W, bits, iters):
    """HQQ：每组「round → 闭式 scale」交替，zero-point 由 min 对齐后固定。
    W:(G,N) float32；bits 在 [2,8]；iters>=0。
    s0=(hi-lo)/(2^bits-1)，z=clamp(round(-lo/s0),0,2^bits-1)；
    迭代：q=clamp(round(W/s)+z,0,2^bits-1)-z，s=<W,q>/<q,q>；
    常量组 scale=1、zero=0、deq=lo。返回 (scale:(G,) float32, zero:(G,) int64, deq:(G,N) float32)。"""
    # TODO
    raise NotImplementedError
