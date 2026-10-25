"""参考答案：Cautious AdamW —— 继承 torch.optim.Optimizer，AdamW 方向 + 谨慎符号掩码。

先按 AdamW 求更新方向 u = (m/bc1)/denom，再套用 apply_cautious 掩码
M = 1[u*g>0] · numel/(sum+1)，最后 p -= lr·(u·M)。复刻 arXiv:2411.16085 的 C-AdamW。
"""

import math

import torch


class _Cautious(torch.optim.Optimizer):
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
                m.mul_(beta1).add_(g, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                bc1 = 1.0 - beta1 ** t
                bc2 = 1.0 - beta2 ** t
                denom = (v.sqrt() / math.sqrt(bc2)).add_(eps)
                update = (m / bc1) / denom  # AdamW 更新方向（未乘 lr）
                mask = (update * g > 0).to(g.dtype)
                mask = mask.mul(mask.numel() / (mask.sum() + 1))  # 谨慎掩码重缩放
                update = update * mask
                p.add_(update, alpha=-lr)
        return loss


def solve(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
    return _Cautious(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
