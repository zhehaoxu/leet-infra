import torch


def _read_seq(sched, n):
    """按「构造即第 0 步、step() 前进」的约定读出 n 个 lr。"""
    lr = sched.optimizer.param_groups[0]["lr"]
    seq = [lr]
    for _ in range(n - 1):
        sched.step()
        seq.append(sched.optimizer.param_groups[0]["lr"])
    return seq


def _oracle(t, base_lr, warmup, stable, decay, p, min_ratio):
    """独立纯 Python 重算 lr(t)：结构化不同的分段实现（先算原始倍率，再统一夹紧）。"""
    if warmup > 0 and t < warmup:
        raw = t / warmup
        gated = True
    elif t < warmup + stable:
        raw, gated = 1.0, False
    elif decay > 0 and t < warmup + stable + decay:
        raw = (1.0 - (t - warmup - stable) / decay) ** p
        gated = True
    else:
        raw, gated = min_ratio, False
    if gated:
        raw = raw * (1.0 - min_ratio) + min_ratio
    return base_lr * max(0.0, raw)


def check_example(solve):
    params = [torch.zeros(2)]
    sched = solve(params, base_lr=0.5, warmup_steps=4, stable_steps=6, decay_steps=5, decay_p=2.0, min_lr_ratio=0.1)
    seq = _read_seq(sched, 15)
    doc = [0.05, 0.1625, 0.275, 0.3875, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.338, 0.212, 0.122, 0.068]
    for a, b in zip(seq, doc):
        assert abs(a - b) < 1e-6, f"示例序列不符: got {seq}"
    for a, b in zip(seq, [_oracle(t, 0.5, 4, 6, 5, 2.0, 0.1) for t in range(15)]):
        assert abs(a - b) < 1e-9, f"示例与独立重算不一致: got {seq}"


check_example.example = True


def check_matches_oracle_polynomial_decay(solve):
    params = [torch.zeros(3)]
    base, warm, stable, decay, p, m = 0.3, 5, 9, 7, 0.5, 0.02
    sched = solve(params, base_lr=base, warmup_steps=warm, stable_steps=stable, decay_steps=decay,
                  decay_p=p, min_lr_ratio=m)
    seq = _read_seq(sched, warm + stable + decay + 4)
    ref = [_oracle(t, base, warm, stable, decay, p, m) for t in range(warm + stable + decay + 4)]
    for i, (a, b) in enumerate(zip(seq, ref)):
        assert abs(a - b) < 1e-9, f"t={i} 偏离独立重算: {a} vs {b}"
    # 三段形态：warmup 严格递增、恒定段相等、退火段单调不增
    assert all(seq[i] < seq[i + 1] for i in range(warm - 1)), "warmup 段应严格递增"
    assert len(set(seq[warm:stable + warm])) == 1, "恒定段 lr 应保持不变"
    tail = seq[warm + stable:warm + stable + decay]
    assert all(tail[i] >= tail[i + 1] - 1e-12 for i in range(len(tail) - 1)), "退火段应单调不增"


def check_matches_transformers_wsd(solve):
    """p=1（线性退火）时与 transformers.get_wsd_schedule(decay_type='linear') 逐步对拍。"""
    from transformers import get_wsd_schedule

    base, warm, stable, decay, m = 0.25, 3, 5, 6, 0.08
    mine_params = [torch.zeros(2)]
    sched = solve(mine_params, base_lr=base, warmup_steps=warm, stable_steps=stable, decay_steps=decay,
                  decay_p=1.0, min_lr_ratio=m)
    mine = _read_seq(sched, warm + stable + decay + 3)

    ref_params = [torch.zeros(2)]
    opt = torch.optim.SGD(ref_params, lr=base)
    ref_sched = get_wsd_schedule(opt, num_warmup_steps=warm, num_decay_steps=decay, num_stable_steps=stable,
                                 warmup_type="linear", decay_type="linear", min_lr_ratio=m)
    ref = [opt.param_groups[0]["lr"]]
    for _ in range(warm + stable + decay + 2):
        ref_sched.step()
        ref.append(opt.param_groups[0]["lr"])
    for i, (a, b) in enumerate(zip(mine, ref)):
        assert abs(a - b) < 1e-12, f"t={i} 与 HF WSD(p=1) 不符: {a} vs {b}"


def check_no_decay_steps_floors_at_min(solve):
    params = [torch.zeros(1)]
    sched = solve(params, base_lr=0.4, warmup_steps=2, stable_steps=3, decay_steps=0, decay_p=2.0, min_lr_ratio=0.25)
    seq = _read_seq(sched, 8)
    assert abs(seq[0] - 0.4 * 0.25) < 1e-12, f"t=0 应为 base_lr·min_lr_ratio，得 {seq}"
    assert abs(seq[2] - 0.4) < 1e-12, f"恒定段应为 base_lr，得 {seq}"
    assert all(abs(v - 0.4 * 0.25) < 1e-12 for v in seq[5:]), f"无退火段时应直接落到下限，得 {seq}"


def check_clamped_beyond_total(solve):
    params = [torch.zeros(1)]
    sched = solve(params, base_lr=0.1, warmup_steps=2, stable_steps=3, decay_steps=4, decay_p=3.0, min_lr_ratio=0.0)
    for _ in range(30):
        sched.step()
    got = sched.optimizer.param_groups[0]["lr"]
    assert abs(got) < 1e-12, f"超出总步数后 lr 应为 0，得 {got}"


def check_returns_scheduler_instance(solve):
    sched = solve([torch.zeros(2)], base_lr=0.1)
    assert isinstance(sched, torch.optim.lr_scheduler.LRScheduler), "solve 须返回 LRScheduler 子类实例"
