import torch


def solve(params, base_lr=0.1, factor=0.5, patience=2, min_lr=0.0, mode="min",
          threshold=1e-4, threshold_mode="rel", cooldown=0, eps=1e-8):
    # TODO: 用 torch.optim.SGD(params, lr=base_lr) 建优化器，返回一个继承
    #   torch.optim.lr_scheduler.LRScheduler 的调度器实例。它没有闭式 lr(t)，而是一台
    #   由外部指标流驱动的状态机：step(metrics) 接收一个可转 float 的指标（标量张量也要能收），
    #   按下面顺序更新 (best, num_bad, cooldown_counter)，只在需要时把每个 param_group 的 lr 乘 factor。
    #     状态初值：best = +inf(mode='min') / -inf(mode='max')，num_bad = 0，cooldown_counter = 0
    #     每次 step(cur)：
    #       1) cur = float(metrics)
    #       2) 若 is_better(cur, best)：best = cur 且 num_bad = 0；否则 num_bad += 1
    #       3) 若 cooldown_counter > 0：cooldown_counter -= 1 且 num_bad = 0（冷却期内的变差全部作废）
    #       4) 若 num_bad > patience（严格大于，等于不降）：降一次 lr，
    #          然后 cooldown_counter = cooldown、num_bad = 0
    #   is_better 用 threshold 过滤"不显著的改善"：
    #     mode='min': rel 时 cur < best*(1-threshold)；abs 时 cur < best - threshold
    #     mode='max': rel 时 cur > best*(1+threshold)；abs 时 cur > best + threshold
    #   降 lr 对每个 param_group：new = max(old*factor, min_lr)，
    #   仅当 old - new > eps 才写回（差值不超过 eps 视为无效降速，保持原 lr）。
    #   step() 不改动 params 本身，只改 param_groups[0]["lr"]。
    # 禁止调用 torch.optim.lr_scheduler.ReduceLROnPlateau 等现成调度器。
    raise NotImplementedError
