"""参考答案：三次渐稀疏率调度（GSL）。"""

import torch


def solve(initial_sparsity, final_sparsity, current_step, first_step, final_step):
    if current_step <= first_step:
        p = initial_sparsity
    elif current_step >= final_step:
        p = final_sparsity
    else:
        frac = (current_step - first_step) / (final_step - first_step)
        p = final_sparsity + (initial_sparsity - final_sparsity) * (1.0 - frac) ** 3
    return torch.tensor(p, dtype=torch.float32)
