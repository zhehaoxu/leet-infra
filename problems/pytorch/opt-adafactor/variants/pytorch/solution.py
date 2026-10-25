"""参考答案：Adafactor —— 继承 torch.optim.Optimizer，实现因子分解二阶矩 + RMS 更新裁剪。

对齐 kozistr/pytorch_optimizer 的 AdaFactor（BigVision 变体、eager per-param、ams_bound=False）：
矩阵参数用行/列统计外积近似二阶矩，更新按 RMS 裁剪，动量 float32、不启用 relative_step。
"""

import math

import torch


def _approx_sq_grad(vr, vc):
    # 行归一化外积重构 v̂_ij = (v_r_i / Σ_k v_r_k) · v_c_j（vr 为末两维的行统计，vc 为列统计）
    r = vr / vr.sum(dim=-1, keepdim=True)
    return r.unsqueeze(-1) * vc


class _Adafactor(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, eps1, eps2, decay_rate, clip_threshold, scale_parameter, weight_decay):
        super().__init__(params, dict(
            lr=lr, betas=betas, eps1=eps1, eps2=eps2, decay_rate=decay_rate,
            clip_threshold=clip_threshold, scale_parameter=scale_parameter, weight_decay=weight_decay,
        ))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for grp in self.param_groups:
            b1, b2cap = grp["betas"]
            if "step" not in grp:
                grp["step"] = 0
            grp["step"] += 1
            t = grp["step"]
            b2t = min(b2cap, 1.0 - math.pow(t, grp["decay_rate"]))
            b = 1.0 - b2t
            for p in grp["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                factored = g.ndim >= 2
                if len(st) == 0:
                    st["exp_avg"] = torch.zeros_like(p)
                    if factored:
                        st["vr"] = torch.zeros(g.shape[:-1], dtype=p.dtype)
                        st["vc"] = torch.zeros(g.shape[-1:], dtype=p.dtype)
                    else:
                        st["v"] = torch.zeros_like(p)

                lr_eff = grp["lr"]
                if grp["scale_parameter"]:
                    rms = p.pow(2).mean().sqrt().item()
                    lr_eff = max(grp["eps2"], rms) * grp["lr"]

                U = g.pow(2).add_(grp["eps1"])
                if factored:
                    st["vr"].lerp_(U.mean(dim=-1), weight=b)
                    st["vc"].lerp_(U.mean(dim=-2), weight=b)
                    inv = torch.rsqrt(_approx_sq_grad(st["vr"], st["vc"]))
                else:
                    st["v"].lerp_(U, weight=b)
                    inv = torch.rsqrt(st["v"])

                u = inv * g
                r_u = u.pow(2).mean().sqrt()
                factor = torch.clamp(r_u / grp["clip_threshold"], min=1.0)
                u = u / factor * lr_eff

                st["exp_avg"].lerp_(u, weight=1 - b1)
                u = st["exp_avg"]

                if grp["weight_decay"] != 0:
                    p.mul_(1 - lr_eff * grp["weight_decay"])
                p.sub_(u)
        return loss


def solve(params, lr=1e-2, betas=(0.9, 0.999), eps1=1e-30, eps2=1e-3, decay_rate=-0.8,
          clip_threshold=1.0, scale_parameter=True, weight_decay=0.0):
    return _Adafactor(params, lr=lr, betas=betas, eps1=eps1, eps2=eps2, decay_rate=decay_rate,
                      clip_threshold=clip_threshold, scale_parameter=scale_parameter, weight_decay=weight_decay)
