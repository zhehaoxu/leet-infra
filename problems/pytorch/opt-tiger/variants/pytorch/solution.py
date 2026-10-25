"""参考答案：Tiger —— 继承 torch.optim.Optimizer，动量符号更新 + 解耦权重衰减。

复刻 kozistr/pytorch-optimizer 的 Tiger（weight_decouple=True）：
仅一份动量 m（初值 0），m ← β·m + (1−β)·g，p ← p − lr·sign(m)，无二阶矩与偏差修正。
"""

import torch


class _Tiger(torch.optim.Optimizer):
    def __init__(self, params, lr, beta, weight_decay):
        super().__init__(params, dict(lr=lr, beta=beta, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, beta, wd = group["lr"], group["beta"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if len(st) == 0:
                    st["exp_avg"] = torch.zeros_like(p)
                m = st["exp_avg"]
                if wd != 0:
                    p.mul_(1 - lr * wd)  # 解耦权重衰减
                m.lerp_(g, weight=1 - beta)
                p.add_(torch.sign(m), alpha=-lr)
        return loss


def solve(params, lr=1e-3, beta=0.965, weight_decay=0.01):
    return _Tiger(params, lr=lr, beta=beta, weight_decay=weight_decay)
