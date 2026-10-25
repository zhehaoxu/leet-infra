"""参考答案：Noam 逆平方根调度（inverse square root）—— 继承 torch.optim.lr_scheduler.LRScheduler。

Vaswani et al. (arXiv:1706.03762, 式 (5))：
    lr(t) = d_model**(-0.5) * min(t**(-0.5), t * warmup_steps**(-1.5))
线性升、平方根降，两支在 t=warmup_steps 处相交（交点即峰值 d**-0.5·w**-0.5），
整个曲线只由"模型宽度"与"warmup 步数"两个量决定，没有总步数。

实现按分段式写，t=0 时取线性支得 0（直接写 min 形式会在 t=0 触发 0**-0.5 的除零）。
约定与 opt-warmup-cosine 一致：构造即第 0 步、step() 才前进，所以不走父类自动记账，
自管 _t，构造时写 lr(0)=0，每次 step() 令 _t+=1 并写 lr(_t)。
"""

import torch


def _lr_at(t, model_dim, warmup_steps):
    coef = model_dim ** -0.5
    if t < warmup_steps:
        return coef * t * warmup_steps ** -1.5
    return coef * t ** -0.5


class _InverseSqrtNoam(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer, model_dim, warmup_steps):
        self.model_dim = model_dim
        self.warmup_steps = warmup_steps
        self.optimizer = optimizer
        self._t = 0
        for g in optimizer.param_groups:
            g["lr"] = _lr_at(0, model_dim, warmup_steps)

    def step(self, epoch=None):
        self._t += 1
        for g in self.optimizer.param_groups:
            g["lr"] = _lr_at(self._t, self.model_dim, self.warmup_steps)


def solve(params, model_dim=512, warmup_steps=4000):
    optimizer = torch.optim.SGD(params, lr=model_dim ** -0.5)
    return _InverseSqrtNoam(optimizer, model_dim=model_dim, warmup_steps=warmup_steps)
