"""参考答案：线性 warmup + 余弦退火调度器 —— 继承 torch.optim.lr_scheduler.LRScheduler。

约定「构造即第 0 步、step() 才前进」，因此不调用父类 __init__ 的自动 step/记账，自管计数 _t：
构造时把优化器 lr 写成 lr(0)，每次 step() 令 _t+=1 并写回 lr(_t)。lr(t) 为闭式分段函数。
"""

import math

import torch


def _lr_at(t, base_lr, warmup_steps, total_steps, min_lr):
    if warmup_steps > 0 and t < warmup_steps:
        return base_lr * (t + 1) / warmup_steps
    denom = total_steps - warmup_steps
    progress = (t - warmup_steps) / denom if denom > 0 else 1.0
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


class _WarmupCosine(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.optimizer = optimizer
        self._t = 0
        # 构造即第 0 步：直接把 lr(0) 写回，不走父类自动 step 记账
        for g, b in zip(optimizer.param_groups, self.base_lrs):
            g["lr"] = _lr_at(0, b, warmup_steps, total_steps, min_lr)

    def step(self, epoch=None):
        self._t += 1
        for g, b in zip(self.optimizer.param_groups, self.base_lrs):
            g["lr"] = _lr_at(self._t, b, self.warmup_steps, self.total_steps, self.min_lr)


def solve(params, base_lr=0.1, warmup_steps=3, total_steps=10, min_lr=0.0):
    optimizer = torch.optim.SGD(params, lr=base_lr)
    return _WarmupCosine(optimizer, warmup_steps=warmup_steps, total_steps=total_steps, min_lr=min_lr)
