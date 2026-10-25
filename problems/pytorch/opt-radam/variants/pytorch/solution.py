"""参考答案：RAdam —— 继承 torch.optim.Optimizer，内建 warmup 的整流自适应更新。

对齐 kozistr/pytorch_optimizer 的 RAdam（degenerated_to_sgd=False, adam_debias=False, per-param 路径）：
显式算有效平均长度 n_t 与稳定系数 r_t；n_t 未达阈值时整步 no-op（连权重衰减都不做）。
"""

import math

import torch


class _RAdam(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay, n_sma_threshold):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, n_sma_threshold=n_sma_threshold))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr, eps, wd, thr = group["lr"], group["eps"], group["weight_decay"], group["n_sma_threshold"]
            n_sma_max = 2.0 / (1.0 - beta2) - 1.0
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
                b2t = beta2 ** t
                n_sma = n_sma_max - 2 * t * b2t / (1.0 - b2t)
                if n_sma >= thr:
                    rt = math.sqrt((1 - b2t) * (n_sma - 4) / (n_sma_max - 4) * (n_sma - 2) / n_sma * n_sma_max / (n_sma_max - 2))
                else:
                    rt = -1.0
                step_size = lr * rt / (1.0 - beta1 ** t)
                if not (step_size > 0 or n_sma >= thr):
                    continue                             # 未达阈值：整步 no-op
                g = p.grad
                m, v = st["exp_avg"], st["exp_avg_sq"]
                if wd != 0:
                    p.mul_(1 - lr * wd)
                m.lerp_(g, 1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                if n_sma >= thr:
                    denom = v.sqrt().add_(eps)
                    p.addcdiv_(m, denom, value=-step_size)
                elif step_size > 0:                       # degenerated_to_sgd 分支（本题不触发）
                    p.add_(m, alpha=-step_size)
        return loss


def solve(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, n_sma_threshold=5):
    return _RAdam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, n_sma_threshold=n_sma_threshold)
