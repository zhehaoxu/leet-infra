"""参考答案：Lion —— 继承 torch.optim.Optimizer，逐参数实现符号动量更新。

对齐 kozistr/pytorch_optimizer 的 Lion（eager 路径、weight_decouple=True）：
先用旧动量合成方向 u=sign(β1·m+(1−β1)·g)，再以 β2 更新动量，最后 p ← p − lr·u。
"""

import torch


class _Lion(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, weight_decay):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr, wd = group["lr"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if len(st) == 0:
                    st["exp_avg"] = torch.zeros_like(p)
                m = st["exp_avg"]
                if wd != 0:
                    p.mul_(1 - lr * wd)  # 解耦权重衰减，先于方向
                u = m.lerp(g, 1 - beta1).sign_()  # 用旧 m
                m.lerp_(g, 1 - beta2)             # 再更新动量
                p.add_(u, alpha=-lr)
        return loss


def solve(params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
    return _Lion(params, lr=lr, betas=betas, weight_decay=weight_decay)
