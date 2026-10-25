import torch


def solve(params, lr=1e-2, betas=(0.9, 0.999), initial_accumulator=1e-6, eps=1e-3, weight_decay=0.0):
    # TODO: 返回一个继承 torch.optim.Optimizer 的 Yogi 实例，step() 就地更新 params。
    #   每参数 m=v=initial_accumulator 初始化（非零）。第 t 步（从 1 起）：
    #     1) 解耦 wd（≠0 先做）：p ← p·(1 − lr·wd)
    #     2) m ← β1·m + (1−β1)·g
    #     3) v ← v − (1−β2)·sign(v − g²)·g²        # Yogi 的符号控制二阶矩
    #     4) bc1 = 1−β1^t ; bc2 = 1−β2^t ; step = lr/bc1
    #     5) p ← p − step · m / (√v/√bc2 + eps)
    #   grad 为 None 的参数整体跳过。禁止调用现成优化器。
    raise NotImplementedError
