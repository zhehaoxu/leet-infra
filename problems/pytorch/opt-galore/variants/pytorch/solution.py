"""参考答案：GaLore 投影梯度优化器 —— 显式给定正交基 P、矩估计在低秩投影空间里做的 AdamW。

口径（与 starter.py 的契约一致）：
- 投影基 P 是入参（形状 [p.numel(), r]），不做 SVD、无随机性；梯度先展平（row-major）
  再投影 g̃ = Pᵀg，一阶/二阶矩、偏差修正与自适应方向全部在 [r] 维投影空间维护，
  最后映回原空间 Δ = P·û 并 p ← p − lr·Δ。proj 为 None 的参数走原空间 AdamW。
- 「基重算」在本设定下是纯整数日程：(t−1) % update_proj_gap == 0 记一次（首步 t=1 建基，
  此后每隔 gap 步一次），写入 state[p]["recompute_steps"]，不影响数值轨迹；
  另暴露纯函数 recompute_at(t)。该日程与上游 GaLoreProjector.project 的
  `ortho_matrix is None or iter % update_proj_gap == 0`（iter 为自增前步数）等价。
- 向量化实现：逐参数用 torch 张量算子（mv/addcmul_/addcdiv_ 风格），与 test_cases 里
  纯 Python 双层循环的独立重算形成结构性对照。
"""

import math

import torch


class _GaLoreAdamW(torch.optim.Optimizer):
    def __init__(self, params, proj, lr, betas, eps, weight_decay, update_proj_gap):
        plist = list(params)
        if len(proj) != len(plist):
            raise ValueError("proj 必须与 params 一一对应")
        super().__init__(plist, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                                     update_proj_gap=update_proj_gap))
        self._proj = {}
        for p, P in zip(plist, proj):
            self._proj[id(p)] = None if P is None else P.detach().clone()

    def recompute_at(self, t):
        """纯布尔日程：第 t 步（从 1 计）是否会重算投影基（上游 `iter % gap == 0` 口径，iter 为
        该参数已完成步数 t−1）。"""
        gap = self.defaults["update_proj_gap"]
        return (t - 1) % gap == 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr, wd, eps = group["lr"], group["weight_decay"], group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                P = self._proj[id(p)]
                st = self.state[p]
                if len(st) == 0:
                    st["step"] = 0
                    r = p.numel() if P is None else P.shape[1]
                    st["exp_avg"] = torch.zeros(r, dtype=p.dtype, device=p.device)
                    st["exp_avg_sq"] = torch.zeros_like(st["exp_avg"])
                    if P is not None:
                        st["recompute_steps"] = []
                st["step"] += 1
                t = st["step"]
                if P is not None and self.recompute_at(t):
                    st["recompute_steps"].append(t)
                if wd != 0:
                    p.mul_(1 - lr * wd)  # 解耦权重衰减，先于矩估计、作用在原空间
                if P is None:
                    gp = g.reshape(-1)
                else:
                    gp = torch.mv(P.t(), g.reshape(-1))  # g̃ = Pᵀ g，形状 [r]
                m, v = st["exp_avg"], st["exp_avg_sq"]
                m.mul_(beta1).add_(gp, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(gp, gp, value=1 - beta2)
                bc1 = 1.0 - beta1 ** t
                bc2 = 1.0 - beta2 ** t
                denom = (v.sqrt() / math.sqrt(bc2)).add_(eps)
                u_hat = torch.div(m, bc1)
                u_hat.div_(denom)
                if P is None:
                    delta = u_hat
                else:
                    delta = torch.mv(P, u_hat)  # 映回原空间
                p.add_(delta.reshape(p.shape), alpha=-lr)
        return loss


def solve(params, proj, lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, update_proj_gap=200):
    return _GaLoreAdamW(params, proj, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        update_proj_gap=update_proj_gap)
