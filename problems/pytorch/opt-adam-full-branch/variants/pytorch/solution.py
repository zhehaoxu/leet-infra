"""参考答案：torch.optim.Adam 全分支账（amsgrad / L2 weight_decay / 多步状态）。

逐行复刻 torch 2.14.1 `torch/optim/adam.py`：`Adam.__init__` 的五道参数校验
（顺序敏感、NaN 也会被 `not 0.0 <= x` 拦下）+ `_single_tensor_adam` 的非
capturable 路径。
"""

import torch


def _check(lr, betas, eps, weight_decay):
    if not 0.0 <= lr:
        raise ValueError(f"Invalid learning rate: {lr}")
    if not 0.0 <= eps:
        raise ValueError(f"Invalid epsilon value: {eps}")
    if not 0.0 <= betas[0] < 1.0:
        raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
    if not 0.0 <= betas[1] < 1.0:
        raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
    if not 0.0 <= weight_decay:
        raise ValueError(f"Invalid weight_decay value: {weight_decay}")


def solve(params, grads, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
          amsgrad=False, maximize=False, state=None):
    betas = tuple(betas)
    if len(betas) != 2:
        raise ValueError(f"betas must have exactly 2 entries, got {len(betas)}")
    _check(lr, betas, eps, weight_decay)
    beta1, beta2 = betas[0], betas[1]

    p = params.clone()
    if state is None:
        m = torch.zeros_like(p)
        v = torch.zeros_like(p)
        mx = torch.zeros_like(p)
        step = 0
    else:
        m = state["exp_avg"].clone()
        v = state["exp_avg_sq"].clone()
        mx = state["max_exp_avg_sq"].clone() if amsgrad else torch.zeros_like(p)
        step = int(state["step"])

    path = [p.clone()]
    steps = []
    deltas = []

    for g_in in grads:
        if g_in is None:
            steps.append(None)
            deltas.append(None)
            path.append(p.clone())
            continue

        step += 1
        g = g_in.clone()
        if maximize:
            g = -g
        if weight_decay != 0:
            g = g + weight_decay * p

        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g * g

        bc1 = 1 - beta1 ** step
        bc2 = 1 - beta2 ** step
        step_size = lr / bc1
        bc2_sqrt = bc2 ** 0.5

        if amsgrad:
            mx = torch.maximum(mx, v)
            denom = mx.sqrt() / bc2_sqrt + eps
        else:
            denom = v.sqrt() / bc2_sqrt + eps

        delta = (m / denom) * (-step_size)
        p = p + delta

        steps.append(step)
        deltas.append(delta.clone())
        path.append(p.clone())

    return {
        "params": p,
        "exp_avg": m,
        "exp_avg_sq": v,
        "max_exp_avg_sq": mx,
        "path": path,
        "steps": steps,
        "deltas": deltas,
    }
