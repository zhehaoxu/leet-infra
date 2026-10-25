"""参考答案：ADOPT —— 继承 torch.optim.Optimizer，去偏差 + 前一步二阶矩 + 裁剪。

复刻 kozistr/pytorch-optimizer 的 ADOPT（clip_lambda=step**0.25）：首步只初始化 v，
其后用上一步 v 归一并裁剪梯度、更新 m 与 p，最后才并入当前 g²；无偏差修正。
"""

import math

import torch


class _ADOPT(torch.optim.Optimizer):
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
                    st["exp_avg_sq"] = torch.zeros_like(p)
                st["step"] += 1
                t = st["step"]
                m, v = st["exp_avg"], st["exp_avg_sq"]
                if wd != 0:
                    p.mul_(1 - lr * wd)  # 解耦权重衰减
                if t == 1:
                    v.addcmul_(g, g)  # 首步仅初始化二阶矩，不更新参数
                    continue
                de_nom = v.sqrt().clamp_(min=eps)  # 用上一步的 v
                normed = g.div(de_nom)
                clip = math.pow(t, 0.25)
                normed = normed.clamp(-clip, clip)
                m.lerp_(normed, weight=1 - beta1)
                p.add_(m, alpha=-lr)  # 无偏差修正
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)  # 最后才并入当前 g²
        return loss


def solve(params, lr=1e-3, betas=(0.9, 0.9999), eps=1e-6, weight_decay=0.0):
    return _ADOPT(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
