import torch


def solve(params, max_lr=0.1, total_steps=10, pct_start=0.3, div_factor=25.0, final_div_factor=1e4,
          anneal_strategy="cos", cycle_momentum=True, base_momentum=0.85, max_momentum=0.95,
          three_phase=False):
    # TODO: 用 torch.optim.SGD(params, lr=max_lr/div_factor, momentum=max_momentum) 建优化器，
    #   返回一个继承 torch.optim.lr_scheduler.LRScheduler 的调度器实例。约定：构造完成即第 0 步，
    #   每调用一次 step() 令 t←t+1，并把 lr(t)（以及 cycle_momentum 时的 momentum(t)）写回
    #   param_groups[0]。记 initial = max_lr/div_factor、floor = initial/final_div_factor：
    #     两阶段（three_phase=False）：
    #       段1 上界 end1 = pct_start*total_steps - 1：lr: initial→max_lr，momentum: max_momentum→base_momentum
    #       段2 上界 end2 = total_steps - 1：        lr: max_lr→floor， momentum: base_momentum→max_momentum
    #     三阶段（three_phase=True）：initial→max_lr、max_lr→initial、initial→floor
    #       （momentum 依次为 max→base、base→max、max→max）
    #   段内比例 pct = (t - 上一段上界) / (本段上界 - 上一段上界)（首段上界为 0.0，越界时夹紧到 [0,1]），
    #   anneal(kind, a, b, pct)：kind='cos' 取 a + (b-a)*(1-cos(pi*pct))/2；kind='linear' 取 a+(b-a)*pct。
    #   lr 与 momentum 用同一个 anneal、同一 pct，但方向相反（lr 升时 momentum 降）。
    # 禁止调用 torch.optim.lr_scheduler.OneCycleLR / SequentialLR 等现成调度器。
    raise NotImplementedError
