import torch


def solve(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
    # TODO: 返回一个继承 torch.optim.Optimizer 的 AdamW 实例，其 step() 就地更新 params。
    #   每个有梯度的参数 p、第 t 步（从 1 起）：
    #     1) 解耦权重衰减（wd≠0 时，先做）：p ← p·(1 − lr·wd)
    #     2) m ← β1·m + (1−β1)·g ; v ← β2·v + (1−β2)·g²
    #     3) bc1 = 1−β1^t ; bc2 = 1−β2^t
    #     4) p ← p − (lr/bc1) · m / (√v/√bc2 + eps)   # 分母算子顺序须与 torch 一致
    #   grad 为 None 的参数整体跳过。禁止调用 torch.optim.AdamW/Adam，需手写。
    raise NotImplementedError
