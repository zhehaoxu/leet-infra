"""参考答案：梯度累积 + 全局范数裁剪 + 动量 SGD —— 继承 torch.optim.Optimizer。

step() 每微批次把 p.grad 累进 a；累计满 accum_steps 个微批次后，先求均值 a/accum、
再按 clip_grad_norm_ 口径算跨参数全局 L2 范数并得共享裁剪系数 c=min(max_norm/(‖ḡ‖+1e-6),1)，
最后走动量 SGD：b←μb+ĝ·c、p←p−lr·b，并清零 a 与微步计数（动量缓冲 b 保留）。
"""

import torch


class _ClipAccum(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum, accum_steps, max_norm):
        super().__init__(params, dict(lr=lr, momentum=momentum, accum_steps=accum_steps, max_norm=max_norm))
        self._micro = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        # 1) 累积
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if len(st) == 0:
                    st["acc"] = torch.zeros_like(p)
                    st["buf"] = torch.zeros_like(p)
                st["acc"].add_(p.grad)
        self._micro += 1
        accum = self.param_groups[0]["accum_steps"]
        if self._micro < accum:
            return loss
        # 2) 全局范数（跨所有参与参数，先取均值）
        sq = torch.zeros(())
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                ge = self.state[p]["acc"] / accum
                sq = sq + ge.pow(2).sum()
        total_norm = sq.sqrt()
        coef = (self.param_groups[0]["max_norm"] / (total_norm + 1e-6)).clamp(max=1.0)
        # 3) 裁剪 + 动量更新 + 复位
        for group in self.param_groups:
            lr = group["lr"]
            mom = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                ge = (st["acc"] / accum) * coef
                st["buf"].mul_(mom).add_(ge)
                p.add_(st["buf"], alpha=-lr)
                st["acc"].zero_()
        self._micro = 0
        return loss


def solve(params, lr, momentum=0.0, accum_steps=1, max_norm=1.0):
    return _ClipAccum(params, lr=lr, momentum=momentum, accum_steps=accum_steps, max_norm=max_norm)
