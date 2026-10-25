"""参考答案：指数滑动平均（EMA）影子权重 + 升温。

EMA 不是 Optimizer：它维护一份与训练参数分离的影子副本，update() 就地读取当前 param 值做
shadow←d·shadow+(1−d)·param。升温 d_n=min(decay,(1+n)/(10+n))（llava/MAE timm EMAHelper 口径），
n 在每次 update() 开头 +1（首次 update 用 n=1）。update() 绝不改动 params。
"""

import torch


class _EMA:
    def __init__(self, params, decay, warmup):
        self.params = list(params)
        self.decay = decay
        self.warmup = warmup
        self.shadow = [p.detach().clone() for p in self.params]
        self._n = 0

    @torch.no_grad()
    def update(self):
        self._n += 1
        if self.warmup:
            d = min(self.decay, (1 + self._n) / (10 + self._n))
        else:
            d = self.decay
        for s, p in zip(self.shadow, self.params):
            s.mul_(d).add_(p.detach(), alpha=1 - d)


def solve(params, decay=0.9, warmup=True):
    return _EMA(params, decay=decay, warmup=warmup)
