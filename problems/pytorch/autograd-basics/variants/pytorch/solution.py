"""参考答案：autograd 求导入门 —— detach 造叶子、backward 读梯度。"""

import torch


def solve(x: torch.Tensor, expr: str) -> torch.Tensor:
    # detach 后新建叶子，避免污染调用方传入的张量
    leaf = x.detach().requires_grad_(True)
    if expr == "quadratic":
        y = 3 * leaf**2 + 2 * leaf
    elif expr == "cubic":
        y = leaf**3
    elif expr == "exp":
        y = torch.exp(leaf)
    else:
        raise ValueError(f"unknown expr: {expr!r}")
    y.sum().backward()
    return leaf.grad
