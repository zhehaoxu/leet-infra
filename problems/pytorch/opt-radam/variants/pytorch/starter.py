import math

import torch


def solve(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, n_sma_threshold=5):
    # TODO: 返回一个继承 torch.optim.Optimizer 的 RAdam 实例，其 step() 就地更新 params。
    #   每个有梯度的参数 p、状态 m, v（初始零），第 t 步（该参数有梯度才自增）：
    #     m ← β1·m + (1−β1)·g ;  v ← β2·v + (1−β2)·g²
    #     n_max = 2/(1−β2) − 1 ;  b2t = β2**t ;  n_t = n_max − 2·t·b2t/(1−b2t)
    #     if n_t >= n_sma_threshold:      # 整流分支，真正更新
    #         r_t = sqrt((1−b2t)·(n_t−4)/(n_max−4)·(n_t−2)/n_t·n_max/(n_max−2))
    #         step = lr·r_t/(1−β1**t)
    #         (wd≠0 时先) p ← p·(1−lr·wd) ;  p ← p − step·m/(sqrt(v)+eps)
    #     else:                            # 未达阈值 → 整步跳过，连 wd 也不做
    #   全程 float32；grad 为 None 的参数整体跳过。禁止调用现成优化器。
    raise NotImplementedError
