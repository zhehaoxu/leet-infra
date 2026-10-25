import math

import torch


def _ref_lr(t, base_lr, warmup_steps, total_steps, min_lr):
    """独立重算 lr(t)：与参考解结构不同（先判 warmup 命中，否则走余弦，显式夹紧）。"""
    if warmup_steps > 0 and t < warmup_steps:
        return base_lr * (t + 1) / warmup_steps
    span = total_steps - warmup_steps
    if span <= 0:
        return min_lr
    frac = (t - warmup_steps) / span
    if frac < 0.0:
        frac = 0.0
    elif frac > 1.0:
        frac = 1.0
    cos_term = 0.5 * (1.0 + math.cos(math.pi * frac))
    return min_lr + (base_lr - min_lr) * cos_term


def _read_seq(sched, n):
    lr = sched.optimizer.param_groups[0]["lr"]
    seq = [lr]
    for _ in range(n - 1):
        sched.step()
        seq.append(sched.optimizer.param_groups[0]["lr"])
    return seq


def check_example(solve):
    params = [torch.zeros(2)]
    sched = solve(params, base_lr=0.1, warmup_steps=3, total_steps=10, min_lr=0.001)
    seq = _read_seq(sched, 10)
    doc = [0.033333, 0.066667, 0.1, 0.1, 0.095098, 0.081363, 0.061515, 0.039485, 0.019637, 0.005902]
    for a, b in zip(seq, doc):
        assert abs(a - b) < 1e-4, f"示例序列不符: got {seq}"
    for a, b in zip(seq, [_ref_lr(t, 0.1, 3, 10, 0.001) for t in range(10)]):
        assert abs(a - b) < 1e-6, f"与参考不一致: got {seq}"


check_example.example = True


def check_pure_cosine_when_no_warmup(solve):
    params = [torch.zeros(1)]
    sched = solve(params, base_lr=0.5, warmup_steps=0, total_steps=8, min_lr=0.0)
    assert abs(sched.optimizer.param_groups[0]["lr"] - 0.5) < 1e-9, "无 warmup 时第 0 步应等于 base_lr"
    seq = _read_seq(sched, 8)
    for a, b in zip(seq, [_ref_lr(t, 0.5, 0, 8, 0.0) for t in range(8)]):
        assert abs(a - b) < 1e-6, f"纯余弦序列不符: {seq}"


def check_matches_oracle_random_hp(solve):
    params = [torch.zeros(3)]
    base, warm, total, mn = 0.3, 4, 25, 0.02
    sched = solve(params, base_lr=base, warmup_steps=warm, total_steps=total, min_lr=mn)
    seq = _read_seq(sched, total)
    ref = [_ref_lr(t, base, warm, total, mn) for t in range(total)]
    for a, b in zip(seq, ref):
        assert abs(a - b) < 1e-6, f"随机超参序列不符: t={seq.index(a)}"
    # 余弦段应单调不增
    cos_part = seq[warm:]
    assert all(cos_part[i] >= cos_part[i + 1] - 1e-12 for i in range(len(cos_part) - 1)), "余弦段须单调不增"


def check_min_lr_clamped_beyond_total(solve):
    params = [torch.zeros(1)]
    sched = solve(params, base_lr=0.1, warmup_steps=2, total_steps=6, min_lr=0.001)
    # 推进到远超 total_steps，progress 夹紧到 1，lr 应恰好等于 min_lr
    for _ in range(30):
        sched.step()
    got = sched.optimizer.param_groups[0]["lr"]
    assert abs(got - 0.001) < 1e-9, f"超出 total 后应夹紧到 min_lr，得 {got}"


def check_returns_scheduler_instance(solve):
    params = [torch.zeros(2)]
    sched = solve(params, base_lr=0.1, warmup_steps=3, total_steps=10, min_lr=0.0)
    assert isinstance(sched, torch.optim.lr_scheduler.LRScheduler), "solve 须返回 LRScheduler 子类实例"
