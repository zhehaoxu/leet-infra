import torch


def solve(params, model_dim=512, warmup_steps=4000):
    # TODO: 用 torch.optim.SGD(params, lr=model_dim**-0.5) 建优化器，返回一个继承
    #   torch.optim.lr_scheduler.LRScheduler 的调度器实例。约定：构造完成即第 0 步，
    #   每调用一次 step() 令 t←t+1 并把 lr(t) 写回 param_groups[0]["lr"]。
    #   Noam（inverse square root）调度：只由 model_dim=d 与 warmup_steps=w 决定，无总步数。
    #     lr(t) = d**(-0.5) * min(t**(-0.5), t * w**(-1.5))     （论文式 (5)，t>=1）
    #   等价的分段写法（t 从 0 计；务必按分段实现，t=0 时不要真的去算 t**(-0.5)，会除零）：
    #     t <  w:  lr = d**(-0.5) * t * w**(-1.5)     # 线性 warmup，lr(0)=0
    #     t >= w:  lr = d**(-0.5) * t**(-0.5)         # 逆平方根衰减，无下限夹紧
    #   两支在 t=w 相交 ⇒ 峰值 lr = d**(-0.5) * w**(-0.5)，且曲线在该点连续。
    #   多参数组时每个组都写同一个 lr(t)（倍率不依赖各组 base_lr）。
    # 禁止调用 get_inverse_sqrt_schedule / LambdaLR 等现成调度器。
    raise NotImplementedError
