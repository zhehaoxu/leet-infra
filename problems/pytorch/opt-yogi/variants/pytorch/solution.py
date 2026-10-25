"""参考答案：Yogi —— 继承 torch.optim.Optimizer，符号控制的二阶矩更新。

复刻 kozistr/pytorch-optimizer 的 Yogi（weight_decouple=True、adam_debias=False）：
状态以 initial_accumulator 非零初始化，v ← v − (1−β2)·sign(v−g²)·g²，其余与 Adam 同。
"""

import math

import torch


class _Yogi(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, initial_accumulator, eps, weight_decay):
        super().__init__(params, dict(lr=lr, betas=betas, initial_accumulator=initial_accumulator,
                                      eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr, ia, eps, wd = group["lr"], group["initial_accumulator"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if len(st) == 0:
                    st["step"] = 0
                    st["exp_avg"] = torch.full_like(p, ia)
                    st["exp_avg_sq"] = torch.full_like(p, ia)
                st["step"] += 1
                t = st["step"]
                m, v = st["exp_avg"], st["exp_avg_sq"]
                if wd != 0:
                    p.mul_(1 - lr * wd)  # 解耦权重衰减
                gp2 = g.mul(g)
                m.lerp_(g, weight=1 - beta1)
                v.addcmul_((v - gp2).sign_(), gp2, value=-(1 - beta2))  # Yogi 二阶矩
                bc2sq = math.sqrt(1.0 - beta2 ** t)
                step_size = lr / (1.0 - beta1 ** t)
                de_nom = v.sqrt().div_(bc2sq).add_(eps)
                p.addcdiv_(m, de_nom, value=-step_size)
        return loss


def solve(params, lr=1e-2, betas=(0.9, 0.999), initial_accumulator=1e-6, eps=1e-3, weight_decay=0.0):
    return _Yogi(params, lr=lr, betas=betas, initial_accumulator=initial_accumulator, eps=eps,
                 weight_decay=weight_decay)
