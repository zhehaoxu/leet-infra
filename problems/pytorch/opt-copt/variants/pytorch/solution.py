"""参考答案：C-Optimizer —— cos(g,m) 连续插值门 + sign 保持截幅的 AdamW 变体。

回源：ByteDance Seed 的 C-Optimizer 思路（论文编号待回源：登记材料里的 2506.01327 经核实
属于另一篇论文；门控为 cos(g,m) 的连续函数 + 幅度控制）；`opt-cautious` 是它的 0/1 离散特例。
本题把连续门定义为 gate = clamp((cos+1)/2, 0, 1)，截幅用逐元素 clamp 到 [−ρ, ρ]（符号保持），
具体映射以 starter docstring 为准。
"""

import torch


class _COpt(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay, rho):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, rho=rho))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            b1, b2 = group["betas"]
            lr, eps, rho, wd = group["lr"], group["eps"], group["rho"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if len(st) == 0:
                    st["step"] = 0
                    st["m"] = torch.zeros_like(p)
                    st["v"] = torch.zeros_like(p)
                st["step"] += 1
                t = st["step"]
                m, v = st["m"], st["v"]
                if wd != 0:
                    p.mul_(1.0 - lr * wd)
                m.mul_(b1).add_(g, alpha=1.0 - b1)
                v.mul_(b2).addcmul_(g, g, value=1.0 - b2)

                gn = g.norm()
                mn = m.norm()
                if float(gn) > 0.0 and float(mn) > 0.0:
                    cos = (g * m).sum() / (gn * mn)
                else:
                    cos = torch.ones((), dtype=g.dtype)
                gate = ((cos + 1.0) * 0.5).clamp(0.0, 1.0)

                mhat = m / (1.0 - b1 ** t)
                vhat = v / (1.0 - b2 ** t)
                u = mhat / (vhat.sqrt() + eps)
                u = u * gate
                u = u.clamp(min=-rho, max=rho)
                p.add_(u, alpha=-lr)
        return loss


def solve(params, lr=0.1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, rho=1.0):
    return _COpt(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, rho=rho)
