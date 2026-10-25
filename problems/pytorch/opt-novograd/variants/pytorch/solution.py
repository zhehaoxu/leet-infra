"""参考答案：NovoGrad —— 继承 torch.optim.Optimizer，逐层标量二阶矩归一 + 纯动量。

复刻 kozistr/pytorch-optimizer 的 NovoGrad（weight_decouple=True、adam_debias=False、无偏差修正）：
二阶矩是每层一个标量 Σg²，首步直接初始化 s 与 m，之后走 EMA/动量递推，步长恒为 lr。
"""

import torch


class _NovoGrad(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay, grad_averaging):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                                      grad_averaging=grad_averaging))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]
            ga = group["grad_averaging"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                st = self.state[p]
                first = len(st) == 0
                g2 = grad.pow(2).sum()  # 整层梯度平方和（标量）
                if first:
                    st["grads_ema"] = g2
                else:
                    st["grads_ema"].lerp_(g2, weight=1 - beta2)
                de_nom = st["grads_ema"].sqrt().add_(eps)
                gn = grad.div(de_nom)
                if wd != 0:
                    p.mul_(1 - lr * wd)  # 解耦权重衰减
                if first:
                    st["moments"] = gn.clone()
                else:
                    if ga:
                        gn = gn.mul(1 - beta1)
                    st["moments"].mul_(beta1).add_(gn)
                p.add_(st["moments"], alpha=-lr)
        return loss


def solve(params, lr=1e-3, betas=(0.95, 0.98), eps=1e-8, weight_decay=0.0, grad_averaging=False):
    return _NovoGrad(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                     grad_averaging=grad_averaging)
