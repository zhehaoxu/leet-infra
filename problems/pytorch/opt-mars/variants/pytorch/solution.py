"""参考答案：MARS（MARS-AdamW 实例化）—— 继承 torch.optim.Optimizer。

对齐论文 arXiv:2411.10438 的式 (3.10)–(3.12) 与 Algorithm 2：方差缩减估计量
c_t = g_t + γ·β1/(1−β1)·(g_t − g_{t−1})（参考实现把 last_grad 零初始化，即首步 g_0 := 0），
再按信任比 ρ_t = min(1, 1/‖c_t‖_F) 做 norm-clip（式 (3.5)），裁剪后的 c̃_t 同时驱动一阶与二阶矩；
可选 ams_bound（参考实现的 AMSGrad 式二阶矩运行最大值，"adaptive clipping of the second moment"）。
偏差修正、解耦权重衰减与分母算子顺序沿用本合集 opt-adamw 的口径。
"""

import math

import torch


class _MARS(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, gamma, eps, weight_decay, ams_bound):
        super().__init__(params, dict(lr=lr, betas=betas, gamma=gamma, eps=eps,
                                      weight_decay=weight_decay, ams_bound=ams_bound))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]
            gamma, ams = group["gamma"], group["ams_bound"]
            gamma_prime = gamma * beta1 / (1.0 - beta1)
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if len(st) == 0:
                    st["step"] = 0
                    st["exp_avg"] = torch.zeros_like(p)
                    st["exp_avg_sq"] = torch.zeros_like(p)
                    st["last_grad"] = torch.zeros_like(p)  # 首步约定 g_0 := 0
                    if ams:
                        st["v_max"] = torch.zeros_like(p)
                st["step"] += 1
                t = st["step"]
                m, v, last = st["exp_avg"], st["exp_avg_sq"], st["last_grad"]
                if wd != 0:
                    p.mul_(1.0 - lr * wd)  # 解耦权重衰减，先于矩估计
                c = (g - last).mul_(gamma_prime).add_(g)  # c = g + γ′·(g − last_g)
                nrm = torch.linalg.vector_norm(c)
                if nrm > 1.0:
                    c = c / nrm  # ρ = 1/‖c‖_F；否则 ρ=1，c̃ = min(1, 1/‖c‖_F)·c
                m.mul_(beta1).add_(c, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(c, c, value=1.0 - beta2)
                base = v
                if ams:
                    torch.maximum(st["v_max"], v, out=st["v_max"])
                    base = st["v_max"]
                denom = base.sqrt().div_(math.sqrt(1.0 - beta2 ** t)).add_(eps)
                p.addcdiv_(m, denom, value=-lr / (1.0 - beta1 ** t))
                last.copy_(g)
        return loss


def solve(params, lr=1e-3, betas=(0.9, 0.99), gamma=0.025, eps=1e-8, weight_decay=0.0, ams_bound=False):
    return _MARS(params, lr=lr, betas=betas, gamma=gamma, eps=eps,
                 weight_decay=weight_decay, ams_bound=ams_bound)
