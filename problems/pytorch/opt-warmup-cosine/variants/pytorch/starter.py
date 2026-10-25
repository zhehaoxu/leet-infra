import torch


def solve(params, base_lr=0.1, warmup_steps=3, total_steps=10, min_lr=0.0):
    # TODO: 用 torch.optim.SGD(params, lr=base_lr) 建优化器，返回一个继承
    #   torch.optim.lr_scheduler.LRScheduler 的调度器实例。约定：构造完成即第 0 步，
    #   优化器 lr 应等于 lr(0)；每调用一次 step() 令 t←t+1 并把 lr(t) 写回
    #   param_groups[0]["lr"]。lr(t) 分段：
    #     warmup 段 (0<=t<warmup_steps): lr = base_lr*(t+1)/warmup_steps
    #     cosine 段 (t>=warmup_steps):   denom = total_steps-warmup_steps
    #        progress = clamp((t-warmup_steps)/denom, 0, 1)  (denom<=0 时取 1)
    #        lr = min_lr + (base_lr-min_lr)*0.5*(1+cos(pi*progress))
    #   禁止调用 LinearLR/CosineAnnealingLR/LambdaLR 等现成调度器。
    raise NotImplementedError
