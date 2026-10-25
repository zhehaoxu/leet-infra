import torch


def _feed(sched, metrics):
    """把固定指标序列喂进状态机，返回每一步之后 param_groups[0]['lr'] 的序列。"""
    out = []
    for m in metrics:
        sched.step(m)
        out.append(sched.optimizer.param_groups[0]["lr"])
    return out


def _native(metrics, base_lr, **kw):
    opt = torch.optim.SGD([torch.zeros(2)], lr=base_lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, **kw)
    return _feed(sched, metrics)


def _mine(solve, metrics, base_lr, **kw):
    return _feed(solve([torch.zeros(2)], base_lr=base_lr, **kw), metrics)


def _oracle(metrics, base_lr, factor, patience, min_lr, mode, threshold, threshold_mode, cooldown, eps):
    """独立纯 Python 重算：不复用调度器代码，自己走一遍同样的状态机（用参考比率/参考差值判改善，
    用条件式而不是 max() 实现地板），返回 lr 序列。"""
    best = float("inf") if mode == "min" else float("-inf")
    lr = float(base_lr)
    bad = 0
    left = 0
    out = []

    def improved(cur):
        if mode == "min":
            ref = best * (1.0 - threshold) if threshold_mode == "rel" else best - threshold
            return cur < ref
        ref = best * (1.0 + threshold) if threshold_mode == "rel" else best + threshold
        return cur > ref

    for m in metrics:
        cur = float(m)
        if improved(cur):
            best = cur
            bad = 0
        else:
            bad = bad + 1
        if left > 0:
            left -= 1
            bad = 0
        if bad > patience:
            cand = lr * factor
            lr = min_lr if cand < min_lr else cand
            bad = 0
            left = cooldown
        out.append(lr)
    return out


def check_example(solve):
    metrics = [1.0, 0.9, 0.95, 0.9, 0.895, 0.7, 0.75, 0.76, 0.77, 0.77, 0.2, 0.3, 0.3, 0.3, 0.3]
    got = _mine(solve, metrics, 0.1, factor=0.5, patience=2, min_lr=0.0, mode="min",
                threshold=1e-4, threshold_mode="rel", cooldown=0, eps=1e-8)
    doc = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.025, 0.025]
    assert got == doc, f"示例序列不符: got {got}"
    ref = _oracle(metrics, 0.1, 0.5, 2, 0.0, "min", 1e-4, "rel", 0, 1e-8)
    assert all(abs(a - b) < 1e-12 for a, b in zip(got, ref)), f"与独立重算不一致: {got} vs {ref}"


check_example.example = True


def check_matches_native_variants(solve):
    """mode × threshold_mode × patience/cooldown/min_lr/eps 的多种组合，与原生逐位对拍。"""
    torch.manual_seed(42)
    cases = [
        (dict(factor=0.5, patience=2, min_lr=0.0, mode="min", threshold=1e-4, threshold_mode="rel",
              cooldown=0, eps=1e-8), [round(float(v), 6) for v in (torch.rand(40) * 0.6 + 0.2)]),
        (dict(factor=0.5, patience=2, min_lr=0.0, mode="max", threshold=1e-4, threshold_mode="rel",
              cooldown=0, eps=1e-8), [round(float(v), 6) for v in (torch.rand(40) * 0.6 + 0.2)]),
        (dict(factor=0.5, patience=2, min_lr=0.0, mode="min", threshold=1e-4, threshold_mode="abs",
              cooldown=0, eps=1e-8), [round(float(v), 6) for v in (torch.rand(40) * 0.6 + 0.2)]),
        (dict(factor=0.7, patience=1, min_lr=0.01, mode="min", threshold=0.01, threshold_mode="abs",
              cooldown=2, eps=1e-8), [round(float(v), 6) for v in (torch.rand(40) * 0.6 + 0.2)]),
        (dict(factor=0.1, patience=0, min_lr=0.0, mode="max", threshold=1e-2, threshold_mode="rel",
              cooldown=3, eps=1e-3), [round(float(v), 6) for v in (torch.rand(40) * 0.6 + 0.2)]),
    ]
    for kw, metrics in cases:
        got = _mine(solve, metrics, 0.1, **kw)
        ref = _native(metrics, 0.1, **kw)
        assert len(got) == len(ref)
        for i, (a, b) in enumerate(zip(got, ref)):
            assert abs(a - b) < 1e-15, f"{kw} 在第 {i} 步与原生不符: {a} vs {b}"


def check_matches_oracle_random_stream(solve):
    """独立重算 + 平台期不变量：lr 单调不增、不越过 min_lr、降速次数与账本一致。"""
    torch.manual_seed(42)
    metrics = [round(float(v), 6) for v in (torch.rand(30) * 0.5 + 0.25)]
    kw = dict(factor=0.6, patience=2, min_lr=0.01, mode="min", threshold=1e-4, threshold_mode="rel",
              cooldown=1, eps=1e-8)
    got = _mine(solve, metrics, 0.3, **kw)
    ref = _oracle(metrics, 0.3, kw["factor"], kw["patience"], kw["min_lr"], kw["mode"], kw["threshold"],
                  kw["threshold_mode"], kw["cooldown"], kw["eps"])
    for i, (a, b) in enumerate(zip(got, ref)):
        assert abs(a - b) < 1e-12, f"第 {i} 步偏离独立重算: {a} vs {b}"
    assert all(got[i] >= got[i + 1] - 1e-15 for i in range(len(got) - 1)), "lr 只能不增"
    assert all(v >= kw["min_lr"] - 1e-15 for v in got), "lr 不得越过 min_lr"
    n_cut = sum(1 for i in range(1, len(got)) if abs(got[i] - got[i - 1]) > 1e-15)
    assert n_cut == 5, f"降速次数应为 5，实际 {n_cut}: {got}"
    assert abs(got[-1] - 0.023328) < 1e-9, f"终点 lr 异常: {got[-1]}"


def check_cooldown_suppresses_bad_steps(solve):
    """冷却期内变差的步数被作废：降一次后要空转 cooldown 步才可能再降。"""
    metrics = [1.0, 0.9, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95]
    got = _mine(solve, metrics, 0.1, factor=0.5, patience=1, min_lr=0.0, mode="min",
                threshold=1e-4, threshold_mode="rel", cooldown=2, eps=1e-8)
    doc = [0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.025]
    assert got == doc, f"冷却序列不符: got {got}"


def check_floor_and_eps_no_op(solve):
    """退化两条：到 min_lr 地板后降速幂等；old-new<=eps 的降速被完全忽略。"""
    floored = _mine(solve, [1.0, 0.9, 0.9, 0.9, 0.9, 0.9], 0.02, factor=0.5, patience=0, min_lr=0.015,
                    mode="min", threshold=1e-4, threshold_mode="rel", cooldown=0, eps=1e-8)
    assert floored[2] == 0.015, f"首次降速应落到 min_lr 而非 0.01: {floored}"
    assert all(v == 0.015 for v in floored[2:]), f"已到底板后应保持不动: {floored}"

    epsed = _mine(solve, [1.0, 0.9, 0.9, 0.9], 0.01, factor=0.5, patience=0, min_lr=0.0, mode="min",
                  threshold=1e-4, threshold_mode="rel", cooldown=0, eps=0.008)
    assert all(v == 0.01 for v in epsed), f"降幅 0.005<=eps=0.008 应被忽略: {epsed}"


def check_first_metric_only_sets_baseline(solve):
    """patience=0 时首指标只建立基线（best=±inf 必然判为更优），第二步步开始才可能降速；
    0 维张量指标同样可用。"""
    got = _mine(solve, [1.0, 0.9, 0.9, 1.5, 1.5], 0.1, factor=0.5, patience=0, min_lr=0.0, mode="min",
                threshold=1e-4, threshold_mode="rel", cooldown=0, eps=1e-8)
    assert got == [0.1, 0.1, 0.05, 0.025, 0.0125], f"patience=0 序列不符: {got}"

    sched = solve([torch.zeros(1)], base_lr=0.1, factor=0.5, patience=0)
    sched.step(torch.tensor(0.5))
    assert sched.optimizer.param_groups[0]["lr"] == 0.1, "0 维张量指标应被正常吃掉且不误降速"


def check_returns_scheduler_instance(solve):
    sched = solve([torch.zeros(2)], base_lr=0.1)
    assert isinstance(sched, torch.optim.lr_scheduler.LRScheduler), "solve 须返回 LRScheduler 子类实例"
    assert sched.optimizer.param_groups[0]["lr"] == 0.1, "构造时不得改动 lr"
