"""参考答案：AdaMuon —— Muon 正交化更新 + 逐元素 RMS 自适应 + 范数还原。

回源：AdaMuon 论文（arXiv:2507.11005；√B 尺度与正交化 O 的逐元素 RMS 交错）与
KellerJordan/Muon 的 zeropower_via_newtonschulz5（quintic (3.4445,-4.7750,2.0315)、5 步、
Frobenius 归一化；bfloat16 迭代改 float32 以稳定 CPU 判题）。与 opt-muon 的分工：
Muon 无坐标自适应；本题在正交化方向上叠加逐元素 RMS 预条件并用范数还原保尺度。
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


class _AdaMuon(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum, ns_steps, betas, eps, weight_decay):
        super().__init__(params, dict(lr=lr, momentum=momentum, ns_steps=ns_steps,
                                      betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            mu = group["momentum"]
            b1, b2 = group["betas"]
            lr, eps, wd, ns = group["lr"], group["eps"], group["weight_decay"], group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if len(st) == 0:
                    st["step"] = 0
                    st["m"] = torch.zeros_like(p)
                    st["v"] = torch.zeros_like(p)
                st["step"] += 1
                t = st["step"]
                if wd != 0:
                    p.mul_(1.0 - lr * wd)
                if p.dim() >= 2:
                    m = st["m"]
                    m.mul_(mu).add_(g, alpha=1.0 - mu)
                    G = g.mul(1.0 - mu).add_(m, alpha=mu)
                    O = _zeropower_newton_schulz(G, ns)
                    v = st["v"]
                    v.mul_(b2).addcmul_(O, O, value=1.0 - b2)
                    vhat = v / (1.0 - b2 ** t)
                    U = O / (vhat.sqrt() + eps)
                    U = U * (O.norm() / (U.norm() + 1e-12))
                    p.add_(U.reshape(p.shape), alpha=-lr)
                else:
                    m, v = st["m"], st["v"]
                    m.mul_(b1).add_(g, alpha=1.0 - b1)
                    v.mul_(b2).addcmul_(g, g, value=1.0 - b2)
                    mhat = m / (1.0 - b1 ** t)
                    vhat = v / (1.0 - b2 ** t)
                    p.add_(mhat / (vhat.sqrt() + eps), alpha=-lr)
        return loss


def solve(params, lr=0.02, momentum=0.95, ns_steps=5, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0):
    return _AdaMuon(params, lr=lr, momentum=momentum, ns_steps=ns_steps, betas=betas,
                    eps=eps, weight_decay=weight_decay)
