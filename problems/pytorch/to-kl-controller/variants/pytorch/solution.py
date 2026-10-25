"""参考答案：自适应 KL 系数控制器（比例控制 + 误差截断）。"""

import torch


def solve(beta, current_kl, target_kl, horizon, n_steps):
    pe = torch.tensor(current_kl / target_kl - 1.0).clamp(-0.2, 0.2)
    mult = 1.0 + pe * (n_steps / horizon)
    return torch.tensor(beta, dtype=torch.float32) * mult
