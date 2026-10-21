"""参考答案：SGD 训练循环与数值复现 —— 严格按题面顺序执行五步。"""

import torch


def solve() -> list[float]:
    w = torch.tensor([2.0], requires_grad=True)
    opt = torch.optim.SGD([w], lr=0.1, momentum=0.9)
    losses: list[float] = []
    for _ in range(20):
        opt.zero_grad()
        loss = (w**2).sum()
        losses.append(loss.item())
        loss.backward()
        opt.step()
    return losses
