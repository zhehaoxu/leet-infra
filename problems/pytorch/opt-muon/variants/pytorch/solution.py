"""参考答案：Muon —— 继承 torch.optim.Optimizer，SGD-动量 + Newton-Schulz 正交化更新。

对齐 KellerJordan/Muon 的 SingleDeviceMuon + zeropower_via_newtonschulz5（quintic 系数
(3.4445,-4.7750,2.0315)、5 步、Frobenius 归一化），并把 bfloat16 迭代改 float32 以稳定 CPU 判题。
"""

import torch


def _zeropower_newton_schulz(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    transposed = False
    if X.size(-2) > X.size(-1):
        X = X.mT
        transposed = True
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X


class _Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum, ns_steps, weight_decay):
        super().__init__(params, dict(lr=lr, momentum=momentum, ns_steps=ns_steps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            mu = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if len(st) == 0:
                    st["momentum_buffer"] = torch.zeros_like(p)
                buf = st["momentum_buffer"]
                buf.mul_(mu).add_(g, alpha=1 - mu)          # buf ← μ·buf + (1−μ)·g
                G = g.mul(1 - mu).add_(buf, alpha=mu)        # nesterov：用更新后的 buf
                O = _zeropower_newton_schulz(G, group["ns_steps"])
                scale = max(1, O.size(-2) / O.size(-1)) ** 0.5
                O = O * scale
                if group["weight_decay"] != 0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(O.reshape(p.shape), alpha=-group["lr"])
        return loss


def solve(params, lr=0.02, momentum=0.95, ns_steps=5, weight_decay=0.0):
    return _Muon(params, lr=lr, momentum=momentum, ns_steps=ns_steps, weight_decay=weight_decay)
