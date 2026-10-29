import torch


def solve(W0, c, alpha, r, base_seed):
    """RandLora 合并：W' = W0 + (alpha/r) * B @ diag(c) @ A。
    B,A 由 base_seed 确定生成；B:(d_out,r), A:(r,d_in), c:(r,)。返回 W':(d_out,d_in)。"""
    d_out, d_in = W0.shape
    g = torch.Generator()
    g.manual_seed(int(base_seed))
    B = torch.randn(d_out, r, generator=g)
    A = torch.randn(r, d_in, generator=g)
    scale = float(alpha) / float(r)
    delta = scale * (B @ torch.diag(c) @ A)
    return W0 + delta
