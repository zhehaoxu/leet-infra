import math

import torch


def solve(params, lr=1e-3, final_lr=0.1, betas=(0.9, 0.999), gamma=1e-3, eps=1e-8, weight_decay=0.0):
    # TODO: 返回一个继承 torch.optim.Optimizer 的 AdaBound 实例，其 step() 就地更新 params。
    #   base_lr = 构造时的 lr。每个有梯度的参数 p、状态 m, v（初始零），第 t 步（有梯度才自增）：
    #     bc1 = 1−β1**t ;  bc2sq = sqrt(1−β2**t)
    #     flr = final_lr*lr/base_lr ;  lower = flr*(1−1/(γt+1)) ;  upper = flr*(1+1/(γt))
    #     step_size = lr*bc2sq/bc1
    #     (wd≠0 时先) p ← p·(1−lr·wd)
    #     m ← β1·m+(1−β1)·g ;  v ← β2·v+(1−β2)·g²
    #     denom = sqrt(v+1e-15)+eps
    #     update = clamp(denom/step_size, lower, upper)*m ;  p ← p − update
    #   全程 float32；grad 为 None 的参数整体跳过。禁止调用现成优化器。
    raise NotImplementedError
