import torch


def solve(params, lr=1e-3, beta=0.965, weight_decay=0.01):
    # TODO: 返回一个继承 torch.optim.Optimizer 的 Tiger 实例，step() 就地更新 params。
    #   每参数一份动量 m（初值 0）。第 t 步、梯度 g：
    #     1) 解耦 wd（≠0 先做）：p ← p·(1 − lr·wd)
    #     2) m ← β·m + (1−β)·g
    #     3) p ← p − lr·sign(m)
    #   无二阶矩、无偏差修正。grad 为 None 的参数整体跳过。禁止调用现成优化器。
    raise NotImplementedError
