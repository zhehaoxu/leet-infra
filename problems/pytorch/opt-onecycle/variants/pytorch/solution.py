"""参考答案：OneCycleLR —— 继承 torch.optim.lr_scheduler.LRScheduler，同时驱动 lr 与 momentum。

严格复刻 torch.optim.lr_scheduler.OneCycleLR 的分段账：
  initial_lr = max_lr/div_factor, min_lr = initial_lr/final_div_factor（可以远低于 base_lr），
  段边界 end_step 用浮点（pct_start*total-1 等），段内 pct=(t-start_step)/(end_step-start_step)，
  lr 与 momentum 在同一段内用同一 anneal 函数但方向相反（up 段 lr↑ mom↓，down 段 lr↓ mom↑）。
约定「构造即第 0 步、step() 才前进」，自管计数 _t，不走父类自动记账。
"""

import math

import torch


def _anneal(kind, start, end, pct):
    if kind == "cos":
        return end + (start - end) / 2.0 * (math.cos(math.pi * pct) + 1)
    return (end - start) * pct + start


class _OneCycle(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer, max_lr, total_steps, pct_start, div_factor, final_div_factor,
                 anneal_strategy, cycle_momentum, base_momentum, max_momentum, three_phase):
        initial_lr = max_lr / div_factor
        min_lr = initial_lr / final_div_factor
        if three_phase:
            self.phases = [
                (float(pct_start * total_steps) - 1.0, initial_lr, max_lr, max_momentum, base_momentum),
                (float(2 * pct_start * total_steps) - 2.0, max_lr, initial_lr, base_momentum, max_momentum),
                (float(total_steps) - 1.0, initial_lr, min_lr, max_momentum, max_momentum),
            ]
        else:
            self.phases = [
                (float(pct_start * total_steps) - 1.0, initial_lr, max_lr, max_momentum, base_momentum),
                (float(total_steps) - 1.0, max_lr, min_lr, base_momentum, max_momentum),
            ]
        self.kind = anneal_strategy
        self.cycle_momentum = cycle_momentum
        self.total_steps = total_steps
        self.optimizer = optimizer
        self._t = 0
        self._write(0)

    def _values(self, t):
        start_step = 0.0
        for i, (end_step, lr0, lr1, m0, m1) in enumerate(self.phases):
            if t <= end_step or i == len(self.phases) - 1:
                pct = (t - start_step) / (end_step - start_step)
                pct = min(max(pct, 0.0), 1.0)  # 只在越界时起作用，段内本就在 [0,1]
                lr = _anneal(self.kind, lr0, lr1, pct)
                mom = _anneal(self.kind, m0, m1, pct) if self.cycle_momentum else None
                return lr, mom
            start_step = end_step
        return self.phases[-1][2], self.phases[-1][4]  # pragma: no cover

    def _write(self, t):
        lr, mom = self._values(t)
        for g in self.optimizer.param_groups:
            g["lr"] = lr
            if mom is not None:
                g["momentum"] = mom

    def step(self, epoch=None):
        self._t += 1
        self._write(self._t)


def solve(params, max_lr=0.1, total_steps=10, pct_start=0.3, div_factor=25.0, final_div_factor=1e4,
          anneal_strategy="cos", cycle_momentum=True, base_momentum=0.85, max_momentum=0.95,
          three_phase=False):
    optimizer = torch.optim.SGD(params, lr=max_lr / div_factor, momentum=max_momentum)
    return _OneCycle(optimizer, max_lr=max_lr, total_steps=total_steps, pct_start=pct_start,
                     div_factor=div_factor, final_div_factor=final_div_factor, anneal_strategy=anneal_strategy,
                     cycle_momentum=cycle_momentum, base_momentum=base_momentum, max_momentum=max_momentum,
                     three_phase=three_phase)
