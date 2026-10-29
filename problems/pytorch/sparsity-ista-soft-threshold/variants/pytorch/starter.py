import torch


def solve(x, W, y, lam, eta):
    """ISTA 一步（L1 近端梯度）：r = W x - y；g = Wᵀ r；z = x - eta·g；
    x_new = sign(z)·max(|z| - lam, 0) 逐元素。
    x float32 (n,)、W float32 (m,n)、y float32 (m,)；lam >= 0、eta >= 0 为标量。
    返回 (x_new, support)：x_new float32 (n,)；support int64 为 |x_new| > 0 的索引（升序）。
    禁止 torch.nn.functional.softshrink / torch.optim. 现成近端算子。"""
    # TODO
    raise NotImplementedError
