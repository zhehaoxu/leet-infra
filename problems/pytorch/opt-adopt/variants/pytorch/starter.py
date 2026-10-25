import torch


def solve(params, lr=1e-3, betas=(0.9, 0.9999), eps=1e-6, weight_decay=0.0):
    # TODO: 返回一个继承 torch.optim.Optimizer 的 ADOPT 实例，step() 就地更新 params。
    #   每参数 m=0, v=0。第 t 步（从 1 起）、梯度 g：
    #     1) 解耦 wd（≠0 先做）：p ← p·(1 − lr·wd)
    #     2) t==1：v ← g²，立即结束本步（不更新 m、p）
    #     3) t≥2：d = clamp(√v, min=eps)；g̃ = clamp(g/d, ±t^0.25)
    #              m ← β1·m + (1−β1)·g̃ ；p ← p − lr·m （无偏差修正）
    #              v ← β2·v + (1−β2)·g² （最后才并入当前梯度平方）
    #   grad 为 None 的参数整体跳过。禁止调用现成优化器。
    raise NotImplementedError
