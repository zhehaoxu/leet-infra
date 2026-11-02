import torch


def solve(params, grads, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
          amsgrad=False, maximize=False, state=None):
    # TODO: 复刻 torch.optim.Adam 的完整分支（多步）。
    #   - 五道校验：lr、eps、betas[0]、betas[1]、weight_decay 都必须满足 `0 <= x`
    #     （betas 还要 `< 1`），顺序固定；NaN 会被 `not 0.0 <= x` 拦下。
    #   - grads 里 None 表示该步无梯度：整步跳过，**不推进 step 计数**（偏差修正的 t
    #     只数真正更新过的步）。
    #   - maximize 先取负，再并 weight_decay（顺序不能反）。
    #   - m = b1*m + (1-b1)*g，v = b2*v + (1-b2)*g*g。
    #   - denom = sqrt(v)/sqrt(1-b2**t) + eps（amsgrad 时用 max(v) 的滑动最大值），
    #     delta = -(lr/(1-b1**t)) * m/denom。
    #   返回 {"params", "exp_avg", "exp_avg_sq", "max_exp_avg_sq", "path", "steps", "deltas"}。
    raise NotImplementedError
