"""参考答案：Sophia-II —— 继承 torch.optim.Optimizer 的裁剪二阶优化器。

Liu et al., *Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training*,
arXiv:2305.14342 的 Algorithm 3（Sophia-II）：
    m_t = β1 m_{t-1} + (1-β1) g_t
    H_t = β2 H_{t-k} + (1-β2) Ĥ^t                （只有估算步才更新，否则 H_t = H_{t-1}）
    θ_t = θ_{t-1}·(1-lr·λ) - lr · clip( m_t / max{H_t, ε}, ρ )

两个最容易被写错的点：
  * 分母是对角 Hessian 的**一次方** max{H, ε}，不是 √H（√H 是 Adam 的口径）；
    所以 Ĥ 整体翻倍 ⇒ 步长精确减半，这条被单独评测。
  * clip 是逐元素夹到 [-ρ, ρ]，负责限幅的是它而不是分母；因此 |Δp| <= lr·ρ 恒成立。

评测把 Hutchinson 估计 Ĥ 当显式入参喂进来：`step(closure=None, hessians=[...])`，
hessians 与 params 同长同序（无梯度的参数位置也要占位）；`hessians=None` 表示本步不做 Hessian
估算 ⇒ H 冻结但仍照常更新 m 与 p，这就是 k 步复用在 CPU 上的确定性表达。
"""

import torch


class _Sophia2(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay, rho):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "rho": rho}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None, hessians=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        index_of = {id(p): i for i, p in enumerate(p for group in self.param_groups for p in group["params"])}
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr, eps, rho, wd = group["lr"], group["eps"], group["rho"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["hessian"] = torch.zeros_like(p)
                state["step"] += 1

                m, h = state["exp_avg"], state["hessian"]
                if wd != 0:
                    p.mul_(1.0 - lr * wd)
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                if hessians is not None:
                    h.mul_(beta2).add_(hessians[index_of[id(p)]], alpha=1.0 - beta2)
                update = (m / h.clamp_min(eps)).clamp_(min=-rho, max=rho)
                p.add_(update, alpha=-lr)

        return loss


def solve(params, lr=0.1, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.01, rho=1.0):
    return _Sophia2(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, rho=rho)
