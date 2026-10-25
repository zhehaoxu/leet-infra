import torch


def _oracle():
    """独立重算：纯 Python 幂次公式 + 手写 ReduceLROnPlateau 状态机。"""
    lr0, gamma = 0.1, 0.5
    lrA = [lr0 * gamma ** i for i in range(4)]
    lrB = [lr0 * gamma ** (i + 1) for i in range(4)]
    wA, wB = 1.0, 1.0
    for lr in lrA:
        wA *= 1.0 - 2.0 * lr          # loss=w^2 -> grad=2w, SGD: w-=lr*2w
    for lr in lrB:
        wB *= 1.0 - 2.0 * lr

    # plateau: mode=min, threshold=1e-4 (rel), factor=0.5, patience=1, min_lr=0.3
    lr, best, num_bad = 1.0, float("inf"), 0
    plateaus = []
    for m in [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]:
        improved = m < best * (1.0 - 1e-4)
        if improved:
            best = min(best, m)
            num_bad = 0
        else:
            num_bad += 1
            if num_bad > 1:
                lr = max(lr * 0.5, 0.3)
                num_bad = 0
        plateaus.append(lr)
    return lrA, lrB, wA, wB, plateaus, best


LR_A, LR_B, W_A, W_B, PLATEAU, BEST = _oracle()


def check_example(solve):
    out = solve()
    assert out["lrs_opt_first"] == LR_A, f"A 账不符: {out['lrs_opt_first']}"
    assert out["lrs_sched_first"] == LR_B, f"B 账不符: {out['lrs_sched_first']}"
    assert out["plateau_lrs"] == PLATEAU, f"plateau 账不符: {out['plateau_lrs']}"
    assert out["last_epoch_A"] == 4 and out["last_epoch_B"] == 4
    assert out["plateau_last_epoch"] == 7 and out["plateau_num_bad"] == 0


check_example.example = True


def check_matches_oracle(solve):
    out = solve()
    assert torch.allclose(torch.tensor(out["w_opt_first"]), torch.tensor(W_A),
                          atol=1e-5, rtol=1e-5), f"w_A 不符: {out['w_opt_first']} vs {W_A}"
    assert torch.allclose(torch.tensor(out["w_sched_first"]), torch.tensor(W_B),
                          atol=1e-5, rtol=1e-5), "w_B 不符"
    assert abs(out["plateau_best"] - BEST) < 1e-9


def check_first_lr_skipped(solve):
    # scheduler 抢跑会「跳过」调度表第一个 lr：B[0] == A[1]
    out = solve()
    A, B = out["lrs_opt_first"], out["lrs_sched_first"]
    assert A[0] == 0.1, "推荐次序下第 0 步应使用完整 base_lr"
    assert B[0] == A[1] and B == A[1:] + [A[3] * 0.5], "抢跑次序整条 lr 前移一格"
    assert out["warned_skipping_first"] is True, "抢跑必须触发官方 UserWarning"


def check_same_last_epoch_different_ledger(solve):
    # 两个变体 last_epoch 都是 4，但实际生效 lr 序列不同、权重终值不同
    out = solve()
    assert out["last_epoch_A"] == out["last_epoch_B"] == 4
    assert out["w_opt_first"] != out["w_sched_first"]
    assert out["w_opt_first"] < out["w_sched_first"], "A 用了更大的首步 lr，应降得更低"


def check_plateau_min_lr_floor(solve):
    # 0.5*0.5=0.25 被 min_lr=0.3 抬底；patience=1 要求连续两个坏 epoch 才降
    out = solve()
    p = out["plateau_lrs"]
    assert p[0] == 1.0 and p[1] == 1.0, "patience=1：第二个坏指标不降"
    assert p[2] == 0.5 and p[4] == 0.3 and p[6] == 0.3
    assert all(x >= 0.3 for x in p), "min_lr 是硬下限"
