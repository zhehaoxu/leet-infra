import torch


def solve(layer_spec):
    """BitFit 参数账：只数 bias（Linear 的输出维 bias；LayerNorm 仅 β）。
    layer_spec:(M,3) int64，每行 [kind,a,b]：0=Linear(a->b,bias=True) 1=LayerNorm(a) 2=Linear(a->b,bias=False)。
    返回 (n_trainable:int, n_total:int, ratio:float)。"""
    n_trainable = 0
    n_total = 0
    for kind, a, b in layer_spec.tolist():
        kind, a, b = int(kind), int(a), int(b)
        if kind == 0:
            n_trainable += b
            n_total += a * b + b
        elif kind == 1:
            n_trainable += a
            n_total += 2 * a
        else:
            n_total += a * b
    return n_trainable, n_total, float(n_trainable) / float(n_total)
