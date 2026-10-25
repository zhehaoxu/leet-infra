import torch


def solve(params, base_lr=0.1, warmup_steps=4, stable_steps=6, decay_steps=5, decay_p=1.0, min_lr_ratio=0.0):
    # TODO: 用 torch.optim.SGD(params, lr=base_lr) 建优化器，返回一个继承
    #   torch.optim.lr_scheduler.LRScheduler 的调度器实例。约定：构造完成即第 0 步，
    #   优化器 lr 应等于 lr(0)；每调用一次 step() 令 t←t+1 并把 lr(t) 写回
    #   param_groups[0]["lr"]。倍率 f(t)（lr(t)=base_lr·f(t)）分四段：
    #     warmup (0<=t<warmup_steps):        f = t/warmup_steps
    #     stable (warmup<=t<warmup+stable):  f = 1
    #     decay  (warmup+stable<=t<warmup+stable+decay):
    #        progress = (t-warmup-stable)/decay_steps
    #        f = (1-progress)**decay_p      # 多项式退火 lr·(1-t)^p
    #     其余（含 decay_steps<=0 时的恒定段之后）: f = min_lr_ratio
    #   warmup 段与 decay 段还要按 min_lr_ratio 线性下移：f ← f·(1−min_lr_ratio)+min_lr_ratio，
    #   并对结果取 max(0, ·)。禁止调用 get_wsd_schedule / LambdaLR 等现成调度器。
    raise NotImplementedError
