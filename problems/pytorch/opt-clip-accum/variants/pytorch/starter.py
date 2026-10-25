import torch


def solve(params, lr, momentum=0.0, accum_steps=1, max_norm=1.0):
    # TODO: 返回一个继承 torch.optim.Optimizer 的实例，step() 就地更新 params。
    #   每参数状态：累积器 a=0、动量缓冲 b=0；实例级微步计数 m=0。每次 step()：
    #     1) 对每个有梯度参数：a += p.grad
    #     2) m += 1；若 m < accum_steps：本次不更新参数，直接返回
    #     3) 否则：
    #          geff = a / accum_steps
    #          total_norm = sqrt( Σ_params ‖geff‖² )      （跨所有参与参数的全局范数）
    #          c = min(max_norm / (total_norm + 1e-6), 1.0)
    #          b = momentum*b + geff*c ;  p -= lr*b
    #          把参与参数的 a 清零；m = 0
    #   grad 为 None 的参数不累积、不参与全局范数、不更新。禁止调用现成优化器/clip_grad_norm_。
    raise NotImplementedError
