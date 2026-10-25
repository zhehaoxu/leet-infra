"""参考答案：scheduler.step() 在循环内的位置、last_epoch 语义与 ReduceLROnPlateau 指标账。"""

import warnings

import torch


def solve():
    # —— 变体 A：optimizer.step() 在前、scheduler.step() 在后（官方推荐次序）
    pA = torch.tensor([1.0], requires_grad=True)
    optA = torch.optim.SGD([pA], lr=0.1)
    schA = torch.optim.lr_scheduler.StepLR(optA, step_size=1, gamma=0.5)
    lrs_opt_first = []
    for _ in range(4):
        lrs_opt_first.append(optA.param_groups[0]["lr"])
        optA.zero_grad()
        (pA * pA).sum().backward()
        optA.step()
        schA.step()

    # —— 变体 B：scheduler.step() 抢在 optimizer.step() 之前
    pB = torch.tensor([1.0], requires_grad=True)
    optB = torch.optim.SGD([pB], lr=0.1)
    schB = torch.optim.lr_scheduler.StepLR(optB, step_size=1, gamma=0.5)
    lrs_sched_first = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(4):
            schB.step()
            lrs_sched_first.append(optB.param_groups[0]["lr"])
            optB.zero_grad()
            (pB * pB).sum().backward()
            optB.step()
    warn_text = " ".join(str(x.message) for x in caught)
    warned_skipping_first = "before" in warn_text and "optimizer.step" in warn_text

    # —— 变体 C：ReduceLROnPlateau，显式喂指标序列
    pC = torch.tensor([1.0], requires_grad=True)
    optC = torch.optim.SGD([pC], lr=1.0)
    schC = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optC, mode="min", factor=0.5, patience=1, min_lr=0.3, threshold=1e-4)
    plateau_lrs = []
    for m in [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]:
        schC.step(m)
        plateau_lrs.append(optC.param_groups[0]["lr"])

    return {
        "lrs_opt_first": lrs_opt_first,
        "lrs_sched_first": lrs_sched_first,
        "w_opt_first": pA.item(),
        "w_sched_first": pB.item(),
        "last_epoch_A": schA.last_epoch,
        "last_epoch_B": schB.last_epoch,
        "warned_skipping_first": warned_skipping_first,
        "plateau_lrs": plateau_lrs,
        "plateau_best": schC.best,
        "plateau_num_bad": schC.num_bad_epochs,
        "plateau_last_epoch": schC.last_epoch,
    }
