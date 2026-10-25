import torch


def solve(params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
    # TODO: 返回一个继承 torch.optim.Optimizer 的 Lion 实例，其 step() 就地更新 params。
    #   每个有梯度的参数 p、动量缓冲 m（初始零）：
    #     1) 解耦权重衰减（wd≠0 时先做）：p ← p·(1 − lr·wd)
    #     2) 用旧 m 求方向：u ← sign(β1·m + (1−β1)·g)
    #     3) 再更新动量：m ← β2·m + (1−β2)·g
    #     4) p ← p − lr·u
    #   注意方向用更新前的 m。grad 为 None 的参数整体跳过。禁止调用现成优化器。
    raise NotImplementedError
