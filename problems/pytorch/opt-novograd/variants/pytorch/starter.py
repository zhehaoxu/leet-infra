import torch


def solve(params, lr=1e-3, betas=(0.95, 0.98), eps=1e-8, weight_decay=0.0, grad_averaging=False):
    # TODO: 返回一个继承 torch.optim.Optimizer 的 NovoGrad 实例，其 step() 就地更新 params。
    #   每个有梯度的参数 p、第 t 步（从 1 起），维护标量层二阶矩 s 与同形动量 m：
    #     1) 层二阶矩：首步 s ← Σᵢ gᵢ² ; 其后 s ← β2·s + (1−β2)·Σᵢ gᵢ²
    #     2) 归一化梯度 ĝ ← g / (√s + eps)
    #     3) 解耦权重衰减（wd≠0）：p ← p·(1 − lr·wd)
    #     4) 动量：首步 m ← ĝ ; 其后（grad_averaging 时先 ĝ←(1−β1)ĝ）m ← β1·m + ĝ
    #     5) p ← p − lr·m
    #   注意：NovoGrad 无偏差修正，步长恒为 lr。grad 为 None 的参数整体跳过。禁止调用现成优化器。
    raise NotImplementedError
