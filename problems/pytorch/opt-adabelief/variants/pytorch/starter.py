import torch


def solve(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-16, weight_decay=0.0):
    # TODO: 返回一个继承 torch.optim.Optimizer 的 AdaBelief 实例，其 step() 就地更新 params。
    #   每个有梯度的参数 p、状态 m, s（初始零），第 t 步（从 1 计）：
    #     1) 解耦权重衰减（wd≠0 时先做）：p ← p·(1 − lr·wd)
    #     2) 一阶动量：m ← β1·m + (1−β1)·g
    #     3) 置信二阶矩（用更新后的 m）：s ← β2·s + (1−β2)·(g−m)² + eps
    #     4) 分母：denom ← sqrt(s)/sqrt(1−β2^t) + eps
    #     5) p ← p − (lr/(1−β1^t)) · m / denom
    #   全程 float32；grad 为 None 的参数整体跳过。禁止调用现成优化器。
    raise NotImplementedError
