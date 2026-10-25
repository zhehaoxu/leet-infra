"""参考答案：Prodigy —— 继承 torch.optim.Optimizer，D-Adaptation 在线估 lr。

复刻 kozistr/pytorch-optimizer 的 Prodigy（beta3=√beta2、bias_correction/safeguard 关闭、
weight_decouple=True、CPU 逐参数两趟路径）：先统计 num/den 并更新 m,v,s，再统一算 d，
最后第二趟用步首旧 d_lr、分母用新 d 更新参数。
"""

import math

import torch


class _Prodigy(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, d0, d_coef, growth_rate, weight_decay, eps):
        super().__init__(params, dict(lr=lr, betas=betas, d=d0, d0=d0, d_max=d0, d_coef=d_coef,
                                      growth_rate=growth_rate, weight_decay=weight_decay, eps=eps))
        self._dnum = None

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        group = self.param_groups[0]
        beta1, beta2 = group["betas"]
        beta3 = math.sqrt(beta2)
        d, d0 = group["d"], group["d0"]
        lr, eps = group["lr"], group["eps"]
        wd, d_coef, gr = group["weight_decay"], group["d_coef"], group["growth_rate"]
        d_lr = d * lr
        dev = group["params"][0].device
        if self._dnum is None:
            self._dnum = torch.tensor([0.0], device=dev)
        d_de_nom = torch.tensor([0.0], device=dev)
        self._dnum.mul_(beta3)  # 先衰减
        # 第一趟：统计 num/den，更新 m, v, s
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            st = self.state[p]
            if len(st) == 0:
                st["s"] = torch.zeros_like(p)
                st["p0"] = p.clone()
                st["exp_avg"] = torch.zeros_like(p)
                st["exp_avg_sq"] = torch.zeros_like(p)
            p0, m, v, s = st["p0"], st["exp_avg"], st["exp_avg_sq"], st["s"]
            self._dnum.add_(torch.dot(grad.flatten(), (p0 - p).flatten()), alpha=(d / d0) * d_lr)
            m.mul_(beta1).add_(grad, alpha=d * (1 - beta1))
            v.mul_(beta2).addcmul_(grad, grad, value=d * d * (1 - beta2))
            s.mul_(beta3).add_(grad, alpha=(d / d0) * d_lr)
            d_de_nom.add_(s.abs().sum())
        if d_de_nom.item() == 0:
            return loss
        d_hat = (d_coef * self._dnum / d_de_nom).item()
        if d == d0:
            d = max(d, d_hat)
        d_max = max(group["d_max"], d_hat)
        d = min(d_max, d * gr)
        group["d"] = d
        group["d_max"] = d_max
        # 第二趟：用步首旧 d_lr 与更新后的新 d 更新参数
        for p in group["params"]:
            if p.grad is None:
                continue
            st = self.state[p]
            m, v = st["exp_avg"], st["exp_avg_sq"]
            if wd != 0:
                p.mul_(1 - d_lr * wd)  # 解耦权重衰减
            de_nom = v.sqrt()
            de_nom.add_(d * eps)
            p.addcdiv_(m, de_nom, value=-d_lr)
        return loss


def solve(params, lr=1.0, betas=(0.9, 0.999), d0=1e-6, d_coef=1.0, growth_rate=float("inf"),
          weight_decay=0.0, eps=1e-8):
    return _Prodigy(params, lr=lr, betas=betas, d0=d0, d_coef=d_coef, growth_rate=growth_rate,
                    weight_decay=weight_decay, eps=eps)
