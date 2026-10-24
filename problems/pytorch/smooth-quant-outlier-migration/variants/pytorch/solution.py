import torch


def solve(X, W, alpha):
    """SmoothQuant：逐输入通道迁移因子 + 迁移后的激活/权重。"""
    ax = X.abs().amax(dim=0)                 # (C,) 激活逐输入通道 absmax
    aw = W.abs().amax(dim=0)                 # (C,) 权重逐输入通道 absmax
    s = ax.pow(alpha) / aw.pow(1 - alpha)
    s = torch.where(s == 0, torch.ones_like(s), s)   # 防除零
    X_hat = X / s.unsqueeze(0)
    W_hat = W * s.unsqueeze(0)
    return s, X_hat, W_hat
