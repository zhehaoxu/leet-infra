"""参考答案：Adam 单步 —— 严格复刻 torch.optim.Adam（amsgrad=False）更新式。"""

import torch


def solve(params, grads, exp_avg, exp_avg_sq, step, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    p = params.clone()
    g = grads.clone()
    m = exp_avg.clone()
    v = exp_avg_sq.clone()
    # 一阶/二阶矩的指数滑动平均
    m = beta1 * m + (1.0 - beta1) * g
    v = beta2 * v + (1.0 - beta2) * g * g
    # 偏差修正（step 为 1-based 当前步）
    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step
    step_size = lr / bc1
    denom = v.sqrt() / (bc2 ** 0.5) + eps
    p = p - step_size * m / denom
    return p, m, v
