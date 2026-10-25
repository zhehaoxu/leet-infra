import torch


def solve(params, base_lr=0.1, T_0=3, T_mult=2, eta_min=0.0):
    # TODO: 用 torch.optim.SGD(params, lr=base_lr) 建优化器，返回一个继承
    #   torch.optim.lr_scheduler.LRScheduler 的调度器实例。约定：构造完成即第 0 步
    #   （此时 lr = base_lr），每调用一次 step() 前进一步并把新的 lr 写回 param_groups[0]["lr"]。
    #   这是 SGDR 余弦暖重启：lr 不是 t 的单调函数，而是"走完一个周期就跳回峰值重开"的
    #   周期账本。状态量 T_cur（本周期内已走的步数）与 T_i（本周期长度），初值 T_cur=0、T_i=T_0。
    #   第 t 步的学习率（η=eta_min、B=该组的 base_lr）：
    #     lr = η + (B - η) * (1 + cos(pi * T_cur / T_i)) / 2
    #   step() 的顺序不可颠倒（先推进账本、再写 lr）：
    #     T_cur ← T_cur + 1
    #     若 T_cur >= T_i：T_cur ← T_cur % T_i，且 T_i ← T_i * T_mult   # 重启：周期逐次拉长
    #     然后按上式写 lr
    #   由此：重启点恰好落在 T_cur=0 处、lr 精确回到 base_lr；而"谷底 η"永远取不到
    #   （T_cur==T_i 那一步在写 lr 之前就已经归零了）。
    #   第 n 个周期（从 0 计）长度为 T_0 * T_mult**n，重启发生在
    #     t = T_0*(T_mult**n - 1)/(T_mult - 1)（T_mult==1 时即 t = n*T_0）。
    #   调度器无终点、不需要 total_steps，可无限续训。禁止调用
    #   CosineAnnealingWarmRestarts / CosineAnnealingLR / SequentialLR 等现成调度器。
    raise NotImplementedError
