import math

import torch


def _read_seq(sched, n):
    """按「构造即第 0 步、step() 前进」的约定读出 n 个 lr。"""
    seq = [sched.optimizer.param_groups[0]["lr"]]
    for _ in range(n - 1):
        sched.step()
        seq.append(sched.optimizer.param_groups[0]["lr"])
    return seq


def _oracle(t, model_dim, warmup_steps):
    """独立重算：直接照论文的 min 闭式写法（t=0 时线性支取 0，避开 0**-0.5 除零）。"""
    coef = model_dim ** -0.5
    if t == 0:
        return 0.0
    return coef * min(t ** -0.5, t * warmup_steps ** -1.5)


def check_example(solve):
    params = [torch.zeros(2)]
    sched = solve(params, model_dim=64, warmup_steps=5)
    seq = _read_seq(sched, 13)
    doc = [0.0, 0.01118034, 0.02236068, 0.03354102, 0.04472136, 0.05590170, 0.05103104,
           0.04724556, 0.04419417, 0.04166667, 0.03952847, 0.03768892, 0.03608439]
    for i, (a, b) in enumerate(zip(seq, doc)):
        assert abs(a - b) < 1e-6, f"t={i} 示例序列不符: got {seq}"
    ref = [_oracle(t, 64, 5) for t in range(13)]
    for i, (a, b) in enumerate(zip(seq, ref)):
        assert abs(a - b) < 1e-12, f"t={i} 与独立重算不符: {a} vs {b}"


check_example.example = True


def check_matches_oracle_min_form(solve):
    """分段实现 vs 论文 min 闭式：多组 (model_dim, warmup_steps) 逐点对拍。"""
    for d, w in [(512, 4000), (4, 1), (9, 3), (100, 7), (64, 5), (2500, 20)]:
        seq = _read_seq(solve([torch.zeros(3)], model_dim=d, warmup_steps=w), 60)
        for t in range(60):
            ref = _oracle(t, d, w)
            assert abs(seq[t] - ref) < 1e-12 * max(1.0, ref), f"d={d} w={w} t={t}: {seq[t]} vs {ref}"


def check_matches_transformers_inverse_sqrt(solve):
    """Noam 与 HF get_inverse_sqrt_schedule 的等价口径：timescale=warmup 时
    HF 的倍率是 (t/warmup 线性升 → sqrt(warmup/t) 降)，乘以峰值 d**-0.5·w**-0.5 即为 Noam。"""
    from transformers import get_inverse_sqrt_schedule

    for d, w in [(64, 5), (16, 4), (9, 1), (121, 9)]:
        n = 40
        mine = _read_seq(solve([torch.zeros(2)], model_dim=d, warmup_steps=w), n)
        peak = d ** -0.5 * w ** -0.5
        opt = torch.optim.SGD([torch.zeros(2)], lr=peak)
        ref_sched = get_inverse_sqrt_schedule(opt, num_warmup_steps=w, timescale=w)
        ref = _read_seq(ref_sched, n)
        assert torch.allclose(torch.tensor(mine), torch.tensor(ref), rtol=1e-12, atol=1e-15), \
            f"d={d} w={w} 与 HF inverse_sqrt 不符:\n mine={mine}\n ref ={ref}"


def check_peak_and_shape_invariants(solve):
    """结构不变量：warmup 段是等差（线性）序列、峰值恰在 t=warmup、其后严格下降、无夹紧。"""
    d, w = 36, 6
    seq = _read_seq(solve([torch.zeros(1)], model_dim=d, warmup_steps=w), w + 30)
    slope = d ** -0.5 * w ** -1.5
    for t in range(1, w):
        assert abs(seq[t] - seq[t - 1] - slope) < 1e-12, f"warmup 段应为等差序列: {seq[:w]}"
    assert abs(seq[w] - d ** -0.5 * w ** -0.5) < 1e-12, f"峰值应为 d**-0.5·w**-0.5: {seq[w]}"
    assert max(seq) == seq[w], f"峰值必须落在 t=warmup: argmax={seq.index(max(seq))}"
    assert all(seq[t] > seq[t + 1] for t in range(w, len(seq) - 1)), "衰减段应严格下降"
    assert all(seq[t] > 0 for t in range(1, len(seq))), "逆平方根衰减不应出现 0 或负值"
    # 两支在 t=w 连续：线性支外推值与衰减支实际值吻合
    assert abs((seq[w] - seq[w - 1]) - slope) < 1e-12, f"t=w 处应与 warmup 斜率连续: {seq[w-1:w+1]}"


def check_no_floor_when_extended(solve):
    """长跑不夹紧：任意 t>=w 都满足 lr(t)·sqrt(t)=d**-0.5，且 t→∞ 仍持续下降。"""
    d, w = 16, 4
    sched = solve([torch.zeros(1)], model_dim=d, warmup_steps=w)
    for _ in range(200):
        sched.step()
    late = sched.optimizer.param_groups[0]["lr"]
    for _ in range(400):
        sched.step()
    later = sched.optimizer.param_groups[0]["lr"]
    assert abs(late * math.sqrt(200) - d ** -0.5) < 1e-9, f"t>=w 后 lr·sqrt(t) 应恒为 d**-0.5，得 {late}"
    assert later < late, f"继续训练应继续下降: {late} -> {later}"
    assert later > 0.0, "不允许被夹紧到 0"


def check_t_zero_is_not_division_by_zero(solve):
    """退化两条：t=0（构造时）lr 必须为 0 且不得因 0**-0.5 崩；w=1 时第 1 步即达峰后纯衰减。"""
    sched = solve([torch.zeros(2)], model_dim=512, warmup_steps=4000)
    assert sched.optimizer.param_groups[0]["lr"] == 0.0, "lr(0) 必须为 0（线性支）"

    seq = _read_seq(solve([torch.zeros(2)], model_dim=4, warmup_steps=1), 6)
    assert abs(seq[0]) < 1e-15 and abs(seq[1] - 0.5) < 1e-12, f"w=1 时 t=1 应达峰 d**-0.5=0.5: {seq}"
    for t in range(1, 6):
        assert abs(seq[t] - 4 ** -0.5 * t ** -0.5) < 1e-12, f"w=1 时 t>=1 为纯逆平方根: {seq}"


def check_returns_scheduler_instance(solve):
    sched = solve([torch.zeros(2)], model_dim=512, warmup_steps=4000)
    assert isinstance(sched, torch.optim.lr_scheduler.LRScheduler), "solve 须返回 LRScheduler 子类实例"
