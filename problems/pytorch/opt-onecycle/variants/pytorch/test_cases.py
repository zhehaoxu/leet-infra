import math

import torch


def _drive(sched, n):
    """按「构造即第 0 步」读出 n 步的 (lr, momentum)。"""
    out = []
    for i in range(n):
        g = sched.optimizer.param_groups[0]
        out.append((g["lr"], g.get("momentum")))
        if i < n - 1:
            sched.step()
    return out


def _oracle(t, max_lr, total, pct_start, div, final_div, kind, cycle_mom, base_m, max_m, three_phase):
    """独立重算：显式列出各段的 (上界, lr 起, lr 止, 动量起, 动量止)，anneal 写法与参考解不同。"""
    initial = max_lr / div
    floor = initial / final_div
    if three_phase:
        segs = [
            (pct_start * total - 1.0, initial, max_lr, max_m, base_m),
            (2 * pct_start * total - 2.0, max_lr, initial, base_m, max_m),
            (total - 1.0, initial, floor, max_m, max_m),
        ]
    else:
        segs = [
            (pct_start * total - 1.0, initial, max_lr, max_m, base_m),
            (total - 1.0, max_lr, floor, base_m, max_m),
        ]

    prev_end = 0.0
    for i, (end, a, b, m0, m1) in enumerate(segs):
        if t <= end or i == len(segs) - 1:
            width = end - prev_end
            frac = 0.0 if width <= 0 else (t - prev_end) / width
            frac = min(max(frac, 0.0), 1.0)
            w = (1.0 - math.cos(math.pi * frac)) / 2.0 if kind == "cos" else frac
            lr = a + (b - a) * w
            mom = m0 + (m1 - m0) * w if cycle_mom else None
            return lr, mom
        prev_end = end
    return initial, max_m if cycle_mom else None  # pragma: no cover



def _native_seq(total, **kw):
    p = [torch.zeros(2)]
    opt = torch.optim.SGD(p, lr=kw["max_lr"] / kw["div_factor"], momentum=kw["max_momentum"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, **kw)
    out = []
    for i in range(total):
        g = opt.param_groups[0]
        out.append((g["lr"], g.get("momentum")))
        if i < total - 1:
            sched.step()
    return out


def check_example(solve):
    kw = dict(max_lr=1.0, total_steps=10, pct_start=0.3, div_factor=10.0, final_div_factor=100.0,
              anneal_strategy="cos", cycle_momentum=True, base_momentum=0.8, max_momentum=0.95,
              three_phase=False)
    seq = _drive(solve([torch.zeros(2)], **kw), 10)
    doc = [(0.1, 0.95), (0.55, 0.875), (1.0, 0.8), (0.950534, 0.807427), (0.811933, 0.828238),
           (0.611649, 0.858311), (0.389351, 0.891689), (0.189067, 0.921762), (0.050466, 0.942573),
           (0.001, 0.95)]
    for i, (lr, mo) in enumerate(seq):
        dlr, dmo = doc[i]
        assert abs(lr - dlr) < 1e-6 and abs(mo - dmo) < 1e-6, f"t={i} 处不符: {seq}"


check_example.example = True


def check_matches_native_all_variants(solve):
    """与原生 torch.optim.lr_scheduler.OneCycleLR 对拍（cos/linear × 两/三阶段 × 是否循环动量）。"""
    for kind in ("cos", "linear"):
        for three_phase in (False, True):
            for cycle_mom in (True, False):
                kw = dict(max_lr=0.5, total_steps=12, pct_start=0.25, div_factor=5.0, final_div_factor=20.0,
                          anneal_strategy=kind, cycle_momentum=cycle_mom, base_momentum=0.8,
                          max_momentum=0.98, three_phase=three_phase)
                mine = _drive(solve([torch.zeros(2)], **kw), 12)
                ref = _native_seq(12, **kw)
                for i, ((a, b), (c, d)) in enumerate(zip(mine, ref)):
                    assert abs(a - c) < 1e-12, f"{kind}/{three_phase}/{cycle_mom} t={i} lr: {a} vs {c}"
                    if cycle_mom:
                        assert abs(b - d) < 1e-12, f"{kind}/{three_phase} t={i} momentum: {b} vs {d}"
                    else:
                        assert b == kw["max_momentum"], f"cycle_momentum=False 时动量不该被改动: {b}"


def check_matches_oracle_random_hp(solve):
    """独立纯 Python 重算 + 反对称不变量：lr 峰值处正是动量谷底。"""
    kw = dict(max_lr=0.24, total_steps=15, pct_start=0.4, div_factor=8.0, final_div_factor=50.0,
              anneal_strategy="cos", cycle_momentum=True, base_momentum=0.7, max_momentum=0.96,
              three_phase=False)
    seq = _drive(solve([torch.zeros(3)], **kw), 15)
    for t, (lr, mom) in enumerate(seq):
        e_lr, e_mom = _oracle(t, kw["max_lr"], kw["total_steps"], kw["pct_start"], kw["div_factor"],
                               kw["final_div_factor"], kw["anneal_strategy"], True, kw["base_momentum"],
                               kw["max_momentum"], False)
        assert abs(lr - e_lr) < 1e-9, f"t={t} lr 偏离独立重算: {lr} vs {e_lr}"
        assert abs(mom - e_mom) < 1e-9, f"t={t} momentum 偏离独立重算: {mom} vs {e_mom}"
    peak = max(range(15), key=lambda i: seq[i][0])
    trough = min(range(15), key=lambda i: seq[i][1])
    assert peak == trough, f"lr 峰值与动量谷底应重合: peak={peak} trough={trough}"
    assert abs(seq[peak][0] - kw["max_lr"]) < 1e-9, "峰值段应命中 max_lr"


def check_final_lr_below_base(solve):
    """OneCycle 的下限 final_div_factor 让终点 lr 低于起点 lr —— 与单调降的余弦调度不同。"""
    kw = dict(max_lr=1.0, total_steps=10, pct_start=0.3, div_factor=10.0, final_div_factor=100.0,
              anneal_strategy="cos", cycle_momentum=True, base_momentum=0.8, max_momentum=0.95,
              three_phase=False)
    seq = _drive(solve([torch.zeros(1)], **kw), 10)
    initial = kw["max_lr"] / kw["div_factor"]
    assert abs(seq[0][0] - initial) < 1e-12, f"起点 lr 应为 max_lr/div_factor，得 {seq[0][0]}"
    assert abs(seq[-1][0] - initial / kw["final_div_factor"]) < 1e-12, f"终点 lr 应为 {initial/100.}"
    assert seq[-1][0] < seq[0][0], "终点必须低于起点（下限可远低于 base_lr）"


def check_momentum_stays_in_band(solve):
    kw = dict(max_lr=0.1, total_steps=10, pct_start=0.5, div_factor=4.0, final_div_factor=10.0,
              anneal_strategy="linear", cycle_momentum=True, base_momentum=0.85, max_momentum=0.95,
              three_phase=True)
    seq = _drive(solve([torch.zeros(1)], **kw), 10)
    for t, (_, mom) in enumerate(seq):
        assert kw["base_momentum"] - 1e-12 <= mom <= kw["max_momentum"] + 1e-12, f"t={t} 动量越界: {mom}"


def check_returns_scheduler_instance(solve):
    sched = solve([torch.zeros(2)], max_lr=0.1, total_steps=6)
    assert isinstance(sched, torch.optim.lr_scheduler.LRScheduler), "solve 须返回 LRScheduler 子类实例"
