import math

import torch


def _read_seq(sched, n):
    """按「构造即第 0 步、step() 前进」的约定读出 n 个 lr。"""
    seq = [sched.optimizer.param_groups[0]["lr"]]
    for _ in range(n - 1):
        sched.step()
        seq.append(sched.optimizer.param_groups[0]["lr"])
    return seq


def _oracle(t, base_lr, T_0, T_mult, eta_min):
    """独立重算：用整数周期账本定位第 t 步落在哪个周期（闭式，不模拟递推状态）。"""
    n = 0
    start = 0
    while True:
        length = T_0 * T_mult ** n
        if t < start + length:
            break
        start += length
        n += 1
    t_cur = t - start
    return eta_min + (base_lr - eta_min) * (1.0 + math.cos(math.pi * t_cur / length)) / 2.0


def _restart_points(seq, base_lr, tol=1e-9):
    return [i for i, v in enumerate(seq) if abs(v - base_lr) < tol]


def check_example(solve):
    params = [torch.zeros(2)]
    sched = solve(params, base_lr=0.1, T_0=3, T_mult=2, eta_min=0.0)
    seq = _read_seq(sched, 14)
    doc = [0.1, 0.075, 0.025, 0.1, 0.09330127, 0.075, 0.05, 0.025, 0.00669873,
           0.1, 0.09829629, 0.09330127, 0.08535534, 0.075]
    for i, (a, b) in enumerate(zip(seq, doc)):
        assert abs(a - b) < 1e-6, f"t={i} 示例序列不符: got {seq}"
    ref = [_oracle(t, 0.1, 3, 2, 0.0) for t in range(14)]
    for i, (a, b) in enumerate(zip(seq, ref)):
        assert abs(a - b) < 1e-12, f"t={i} 与独立重算不符: {a} vs {b}"


check_example.example = True


def check_matches_native_variants(solve):
    """与原生 CosineAnnealingWarmRestarts 逐步对拍，覆盖 T_mult=1/2/3、T_0=1、eta_min 非零。"""
    configs = [(0.1, 3, 2, 0.0), (0.1, 4, 1, 0.0), (0.05, 5, 2, 0.001), (0.2, 2, 3, 0.01),
               (1.0, 1, 2, 0.0), (0.3, 6, 2, 0.05), (0.07, 7, 1, 0.0123)]
    for base, T0, Tm, em in configs:
        mine = _read_seq(solve([torch.zeros(2)], base_lr=base, T_0=T0, T_mult=Tm, eta_min=em), 30)
        opt = torch.optim.SGD([torch.zeros(2)], lr=base)
        native = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T0, Tm, em)
        ref = _read_seq(native, 30)
        for i, (a, b) in enumerate(zip(mine, ref)):
            assert abs(a - b) < 1e-12, f"base={base} T_0={T0} T_mult={Tm} 第 {i} 步与原生不符: {a} vs {b}"


def check_matches_oracle_closed_form(solve):
    """闭式周期账本重算：任意 t 落在第 n 个长度 T_0·T_mult^n 的周期里。"""
    for base, T0, Tm, em in [(0.4, 3, 2, 0.02), (0.09, 5, 1, 0.0), (0.25, 2, 4, 0.03)]:
        seq = _read_seq(solve([torch.zeros(3)], base_lr=base, T_0=T0, T_mult=Tm, eta_min=em), 60)
        for t in range(60):
            assert abs(seq[t] - _oracle(t, base, T0, Tm, em)) < 1e-12, \
                f"T_0={T0} T_mult={Tm} t={t} 偏离闭式重算: {seq[t]}"


def check_restart_ledger(solve):
    """重启点账本：lr 精确回到 base_lr 的位置必须是 T_0·(T_mult^n−1)/(T_mult−1)，间隔逐次 ×T_mult。"""
    base, T0, Tm = 0.5, 4, 2
    seq = _read_seq(solve([torch.zeros(2)], base_lr=base, T_0=T0, T_mult=Tm, eta_min=0.02), 29)
    got = _restart_points(seq, base)
    assert got == [0, 4, 12, 28], f"重启点应为 [0, 4, 12, 28]，实际 {got}"
    gaps = [got[i + 1] - got[i] for i in range(len(got) - 1)]
    assert gaps == [T0 * Tm ** i for i in range(len(gaps))], f"周期长度账本不符: {gaps}"


def check_trough_never_reached(solve):
    """写 lr 之前先重启 ⇒ 谷底 eta_min 永远取不到；每个周期内 lr 严格单调下降。"""
    base, T0, Tm, em = 0.2, 2, 3, 0.01
    seq = _read_seq(solve([torch.zeros(2)], base_lr=base, T_0=T0, T_mult=Tm, eta_min=em), 40)
    assert min(seq) > em + 1e-9, f"谷底 eta_min 不该被取到，实得 min={min(seq)} (eta_min={em})"
    assert all(v >= em for v in seq), f"lr 不得越过 eta_min: {seq}"
    bounds = [0, 2, 8, 26, 40]
    for lo, hi in zip(bounds, bounds[1:]):
        seg = seq[lo:min(hi, 40)]
        assert all(seg[i] > seg[i + 1] for i in range(len(seg) - 1)), f"[{lo},{hi}) 段内应严格下降: {seq}"


def check_fixed_period_when_mult_one(solve):
    """T_mult=1 退化：周期恒为 T_0，重启点为 T_0 的整数倍，且可一直续训不夹紧。"""
    base, T0 = 0.16, 5
    seq = _read_seq(solve([torch.zeros(1)], base_lr=base, T_0=T0, T_mult=1, eta_min=0.0), 26)
    assert _restart_points(seq, base) == [0, 5, 10, 15, 20, 25], f"固定周期重启点不符: {seq}"
    assert abs(seq[24] - 0.015278640450004212) < 1e-12, f"谷底应为 T_cur=T_0-1 处: {seq[24]}"
    assert abs(min(seq) - seq[24]) < 1e-15, f"整段最小值即每周期谷底: {seq}"


def check_returns_scheduler_instance(solve):
    sched = solve([torch.zeros(2)], base_lr=0.1)
    assert isinstance(sched, torch.optim.lr_scheduler.LRScheduler), "solve 须返回 LRScheduler 子类实例"
    assert abs(sched.optimizer.param_groups[0]["lr"] - 0.1) < 1e-12, "构造即第 0 步，lr 应为 base_lr"
