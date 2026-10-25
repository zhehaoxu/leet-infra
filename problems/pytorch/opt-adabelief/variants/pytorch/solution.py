"""参考答案：AdaBelief —— 继承 torch.optim.Optimizer，用 (g−m)² 作置信二阶矩。

对齐 kozistr/pytorch_optimizer 的 AdaBelief（rectify=False, ams_bound=False, per-param eager 路径）：
先解耦权重衰减，再更新一阶动量 m，用「更新后的 m」算残差 (g−m) 累积二阶矩 s，分母带 sqrt(1−β2^t) 修正。
"""

import math

import torch


class _AdaBelief(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if len(st) == 0:
                    st["step"] = 0
                    st["exp_avg"] = torch.zeros_like(p)
                    st["exp_avg_var"] = torch.zeros_like(p)
                st["step"] += 1
                t = st["step"]
                m, s = st["exp_avg"], st["exp_avg_var"]
                if wd != 0:
                    p.mul_(1 - lr * wd)                     # 解耦权重衰减，先于方向
                m.lerp_(g, 1 - beta1)                       # m ← β1·m + (1−β1)·g
                diff = g - m                                # 用更新后的 m
                s.mul_(beta2).addcmul_(diff, diff, value=1 - beta2).add_(eps)
                bc2sq = math.sqrt(1.0 - beta2 ** t)
                step_size = lr / (1.0 - beta1 ** t)
                denom = s.add(0.0).sqrt_().div_(bc2sq).add_(eps)
                p.addcdiv_(m, denom, value=-step_size)
        return loss


def solve(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-16, weight_decay=0.0):
    return _AdaBelief(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
