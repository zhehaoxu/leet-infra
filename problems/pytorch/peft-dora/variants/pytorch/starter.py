import torch


def solve(x, W0, A, B, m, alpha, r):
    """DoRA 前向：V=W0+(alpha/r)*B@A; n=V.norm(dim=1,keepdim=True);
    Weff=(V/n)*m.unsqueeze(1); y=x@Weff.T。
    x:(...,d_in) W0:(d_out,d_in) A:(r,d_in) B:(d_out,r) m:(d_out,)。返回 (...,d_out)。禁止现成 PEFT 库。"""
    # TODO
    raise NotImplementedError
