"""参考答案：WSD（Warmup-Stable-Decay）调度器 —— 继承 torch.optim.lr_scheduler.LRScheduler。

三段式：线性 warmup → 恒定段（stable）→ 末端多项式退火 lr·(1−progress)**decay_p，
退火与 warmup 都按 min_lr_ratio 做线性下移（factor = factor*(1−m)+m），超出总步数后恒为 min_lr_ratio·base_lr。
约定与 opt-warmup-cosine 一致：「构造即第 0 步、step() 才前进」，因此不走父类的自动记账，
自管计数 _t，构造时把 lr(0) 写回、每次 step() 令 _t+=1 并写回 lr(_t)。
"""

import torch


def _lr_ratio(t, warmup_steps, stable_steps, decay_steps, decay_p, min_lr_ratio):
    """返回 lr(base_lr) 的倍率 f(t)：warmup / stable / decay / floor 四段。"""

    def rescale(f):
        return max(0.0, f * (1.0 - min_lr_ratio) + min_lr_ratio)

    if warmup_steps > 0 and t < warmup_steps:
        return rescale(t / warmup_steps)
    if t < warmup_steps + stable_steps:
        return 1.0
    if decay_steps > 0 and t < warmup_steps + stable_steps + decay_steps:
        progress = (t - warmup_steps - stable_steps) / decay_steps
        return rescale((1.0 - progress) ** decay_p)
    return min_lr_ratio


class _WSD(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer, warmup_steps, stable_steps, decay_steps, decay_p, min_lr_ratio):
        self.warmup_steps = warmup_steps
        self.stable_steps = stable_steps
        self.decay_steps = decay_steps
        self.decay_p = decay_p
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.optimizer = optimizer
        self._t = 0
        for g, b in zip(optimizer.param_groups, self.base_lrs):
            g["lr"] = b * _lr_ratio(0, warmup_steps, stable_steps, decay_steps, decay_p, min_lr_ratio)

    def step(self, epoch=None):
        self._t += 1
        for g, b in zip(self.optimizer.param_groups, self.base_lrs):
            g["lr"] = b * _lr_ratio(self._t, self.warmup_steps, self.stable_steps, self.decay_steps,
                                    self.decay_p, self.min_lr_ratio)


def solve(params, base_lr=0.1, warmup_steps=4, stable_steps=6, decay_steps=5, decay_p=1.0, min_lr_ratio=0.0):
    optimizer = torch.optim.SGD(params, lr=base_lr)
    return _WSD(optimizer, warmup_steps=warmup_steps, stable_steps=stable_steps, decay_steps=decay_steps,
                decay_p=decay_p, min_lr_ratio=min_lr_ratio)
