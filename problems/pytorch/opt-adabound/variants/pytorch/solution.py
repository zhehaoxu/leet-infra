"""参考答案：AdaBound —— 继承 torch.optim.Optimizer，给逐坐标有效学习率套动态上下界。

对齐 kozistr/pytorch_optimizer 的 AdaBound（ams_bound=False, per-param 路径）：
把 denom/step_size 夹在随 t 收敛到 final_lr 的 [lower, upper] 内，再乘 m 做更新，实现 Adam→SGD 的过渡。
"""

import math

import torch


class _AdaBound(torch.optim.Optimizer):
    def __init__(self, params, lr, final_lr, betas, gamma, eps, weight_decay):
        super().__init__(params, dict(lr=lr, final_lr=final_lr, betas=betas, gamma=gamma, eps=eps, weight_decay=weight_decay))
        self.base_lrs = [group["lr"] for group in self.param_groups]

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for gi, group in enumerate(self.param_groups):
            beta1, beta2 = group["betas"]
            lr, eps, wd, gamma = group["lr"], group["eps"], group["weight_decay"], group["gamma"]
            base_lr = self.base_lrs[gi]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if len(st) == 0:
                    st["step"] = 0
                    st["exp_avg"] = torch.zeros_like(p)
                    st["exp_avg_sq"] = torch.zeros_like(p)
                st["step"] += 1
                t = st["step"]
                bc1 = 1.0 - beta1 ** t
                bc2sq = math.sqrt(1.0 - beta2 ** t)
                flr = group["final_lr"] * lr / base_lr if base_lr > 0 else 0.0
                lower = flr * (1 - 1 / (gamma * t + 1))
                upper = flr * (1 + 1 / (gamma * t))
                step_size = lr * bc2sq / bc1
                g = p.grad
                m, v = st["exp_avg"], st["exp_avg_sq"]
                if wd != 0:
                    p.mul_(1 - lr * wd)
                m.lerp_(g, 1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                de_nom = v.add(1e-15).sqrt_().add_(eps)
                update = de_nom.div_(step_size)
                update.clamp_(min=lower, max=upper).mul_(m)
                p.sub_(update)
        return loss


def solve(params, lr=1e-3, final_lr=0.1, betas=(0.9, 0.999), gamma=1e-3, eps=1e-8, weight_decay=0.0):
    return _AdaBound(params, lr=lr, final_lr=final_lr, betas=betas, gamma=gamma, eps=eps, weight_decay=weight_decay)
