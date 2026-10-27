import torch


def solve(layer_spec):
    """BitFit 参数账：只数 bias（Linear 的输出维 bias；LayerNorm 仅 β）。
    layer_spec:(M,3) int64，每行 [kind,a,b]：0=Linear(a->b,bias=True) 1=LayerNorm(a) 2=Linear(a->b,bias=False)。
    返回 (n_trainable:int, n_total:int, ratio:float)。"""
    # TODO
    raise NotImplementedError
