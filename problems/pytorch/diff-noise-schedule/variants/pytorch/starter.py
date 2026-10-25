import torch


def solve(T, kind):
    """噪声调度 -> alpha_bar 序列 (T,)。
    kind=='linear': betas=linspace(1e-4,0.02,T)
    kind=='cosine': s=0.008; f(u)=cos(((u/T)+s)/(1+s)*pi/2)**2; ac=f(0..T)/f(0);
                    betas=clip(1-ac[1:]/ac[:-1],0,0.999)
    alpha_bar=cumprod(1-betas)。float32。禁止现成扩散库。"""
    # TODO
    raise NotImplementedError
