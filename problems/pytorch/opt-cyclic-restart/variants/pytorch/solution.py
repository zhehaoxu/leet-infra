"""参考答案：SGDR 余弦暖重启 —— 继承 torch.optim.lr_scheduler.LRScheduler。

Loshchilov & Hutter (arXiv:1608.03983) 的 warm restarts：余弦退火走满一个周期后不复用周期，
而是把 lr **跳回峰值**重开一条余弦，且第 i 个周期长度 T_i = T_0·T_mult^i 逐次拉长。
与原生 CosineAnnealingWarmRestarts 一致的两处关键口径：
  1) 重启判定发生在写 lr **之前**（T_cur 先 +1，若 T_cur>=T_i 就 T_cur%=T_i、T_i*=T_mult），
     所以 T_cur==T_i 那个"谷底 eta_min"永远取不到，每个周期最后一步之后直接是峰值；
  2) 构造完成即第 0 步（此时 T_cur=0、lr=base_lr），每调用一次 step() 前进一步，
     因此不走父类自动记账，自管 _t。
"""

import math

import torch


class _WarmRestarts(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer, T_0, T_mult, eta_min):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_i = T_0
        self.T_cur = 0
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.optimizer = optimizer
        self._t = 0
        self._write()

    def _write(self):
        for g, b in zip(self.optimizer.param_groups, self.base_lrs):
            g["lr"] = self.eta_min + (b - self.eta_min) * (1.0 + math.cos(math.pi * self.T_cur / self.T_i)) / 2.0

    def step(self, epoch=None):
        self._t += 1
        self.T_cur += 1
        if self.T_cur >= self.T_i:
            self.T_cur %= self.T_i
            self.T_i *= self.T_mult
        self._write()


def solve(params, base_lr=0.1, T_0=3, T_mult=2, eta_min=0.0):
    optimizer = torch.optim.SGD(params, lr=base_lr)
    return _WarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min)
