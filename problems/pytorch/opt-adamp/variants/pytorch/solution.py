"""参考答案：AdamP —— 继承 torch.optim.Optimizer，剔除尺度不变参数上的径向更新分量。

对齐 kozistr/pytorch_optimizer 的 AdamP（nesterov=False, cautious=False, per-param 路径）：
按 channel/layer 两个视图检测梯度与参数是否近乎正交，命中则从 Adam 更新向量里减去平行于参数的
径向分量，并把该组解耦权重衰减乘以 wd_ratio。
"""

import math

import torch
from torch.nn.functional import cosine_similarity


def _channel_view(x):
    return x.view(x.size()[0], -1)


def _layer_view(x):
    return x.view(1, -1)


def _cos_by_view(x, y, eps, view_func):
    x = view_func(x)
    y = view_func(y)
    return cosine_similarity(x, y, dim=1, eps=eps).abs_()


def _projection(p, grad, perturb, delta, wd_ratio, eps):
    wd = 1.0
    expand_size = [-1] + [1] * (len(p.shape) - 1)
    for view_func in (_channel_view, _layer_view):
        cosine_sim = _cos_by_view(grad, p, eps, view_func)
        if cosine_sim.max() < delta / math.sqrt(view_func(p).size()[1]):
            p_n = p / view_func(p).norm(dim=1).view(expand_size).add_(eps)
            radial = view_func(p_n * perturb).sum(dim=1).view(expand_size)
            perturb.sub_(p_n.mul_(radial))
            wd = wd_ratio
            return perturb, wd
    return perturb, wd


class _AdamP(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay, delta, wd_ratio):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, delta=delta, wd_ratio=wd_ratio))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]
            delta, wd_ratio = group["delta"], group["wd_ratio"]
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
                bc2sq = math.sqrt(1.0 - beta2 ** t)
                step_size = lr / (1.0 - beta1 ** t)
                m.lerp_(g, 1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                inv_de_nom = v.sqrt().add_(eps).reciprocal_().mul_(bc2sq)
                perturb = inv_de_nom.mul_(m)
                ratio = 1.0
                if len(p.shape) > 1:
                    perturb, ratio = _projection(p, g, perturb, delta, wd_ratio, eps)
                if wd != 0:
                    p.mul_(1 - lr * wd * ratio)
                p.add_(perturb, alpha=-step_size)
        return loss


def solve(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, delta=0.1, wd_ratio=0.1):
    return _AdamP(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, delta=delta, wd_ratio=wd_ratio)
