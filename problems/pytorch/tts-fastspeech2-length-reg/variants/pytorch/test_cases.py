import bisect
import math

from collections.abc import Callable

H1 = [math.log(3), math.log(1)]
FEATS = [0.5, -0.5]


def _linspace(lo, hi, num):
    if num == 1:
        return [lo]
    return [lo + i * (hi - lo) / (num - 1) for i in range(num)]


def _oracle(log_durations, duration_target, d_control,
            expand_features, feat_value, p_control,
            feat_min, feat_max, n_bins, log_scale):
    T = len(log_durations)
    if duration_target is not None:
        rounded = [int(d) for d in duration_target]
    else:
        rounded = [max(round(math.exp(ld) - 1.0) * d_control, 0) for ld in log_durations]
    idx = []
    for i, d in enumerate(rounded):
        idx += [i] * max(int(d), 0)
    if log_scale:
        bins = [math.exp(x) for x in _linspace(math.log(feat_min), math.log(feat_max), n_bins - 1)]
    else:
        bins = _linspace(feat_min, feat_max, n_bins - 1)
    return {
        "duration_rounded": rounded,
        "mel_len": len(idx),
        "expand_indices": idx,
        "feat_bucket": bisect.bisect_left(bins, feat_value * p_control),
    }


def _compare(solve: Callable, *args):
    led = solve(*args)
    exp = _oracle(*args)
    assert led == exp, f"\nled={led}\nexp={exp}"
    return led


def check_basic_duration_chain(solve):
    # round(exp(log3)-1)=2, round(exp(0)-1)=0
    led = _compare(solve, H1, None, 1.0, FEATS, 150.0, 1.0, 100.0, 300.0, 4, False)
    assert led["duration_rounded"] == [2, 0]
    assert led["mel_len"] == 2
    assert led["expand_indices"] == [0, 0]
    assert led["feat_bucket"] == 1  # 150 ∈ [100, 200)


def check_minus_one_offset_and_clamp(solve):
    # exp(log(0.5))-1 = -0.5 → round=0（先减 1 再 round，负值钳 0）
    led = _compare(solve, [math.log(0.5)], None, 1.0, [1.0], 0.0, 1.0, 1.0, 2.0, 3, False)
    assert led["duration_rounded"] == [0]
    assert led["mel_len"] == 0 and led["expand_indices"] == []
    # 极负 logd：exp-1 ≈ -1 → round=-1 → clamp 0
    led2 = _compare(solve, [-5.0], None, 1.0, [1.0], 0.0, 1.0, 1.0, 2.0, 3, False)
    assert led2["duration_rounded"] == [0]


def check_bankers_rounding(solve):
    # exp(logd)-1 = 2.5 → round 银行家 = 2；3.5 → 4
    led = _compare(solve, [math.log(3.5)], None, 1.0, [1.0], 0.0, 1.0, 1.0, 2.0, 2, False)
    assert led["duration_rounded"] == [2]
    led2 = _compare(solve, [math.log(4.5)], None, 1.0, [1.0], 0.0, 1.0, 1.0, 2.0, 2, False)
    assert led2["duration_rounded"] == [4]
    led3 = _compare(solve, [math.log(1.5)], None, 1.0, [1.0], 0.0, 1.0, 1.0, 2.0, 2, False)
    assert led3["duration_rounded"] == [0]  # round(0.5)=0


def check_d_control_truncation(solve):
    # round(2)*1.5 = 3.0 整数；round(2)*0.4 = 0.8 → int 截断 0
    led = _compare(solve, [math.log(3), math.log(4)], None, 0.4, [1.0, 1.0], 0.0, 1.0, 1.0, 2.0, 2, False)
    assert led["duration_rounded"] == [0.8, 1.2000000000000002]
    assert led["expand_indices"] == [1]  # 0.8→0 帧, 1.2→1 帧
    assert led["mel_len"] == 1
    led2 = _compare(solve, [math.log(3)], None, 1.5, [1.0], 0.0, 1.0, 1.0, 2.0, 2, False)
    assert led2["expand_indices"] == [0, 0, 0]  # 3.0 → 3 帧


def check_duration_target_passthrough(solve):
    # 训练分支：duration_target 原样，不经过 round/控制链
    led = _compare(solve, H1, [3, 1], 0.5, FEATS, 0.0, 1.0, 1.0, 2.0, 2, False)
    assert led["duration_rounded"] == [3, 1]
    assert led["expand_indices"] == [0, 0, 0, 1]
    assert led["mel_len"] == 4


def check_bucketize_equal_boundary(solve):
    # right=False = bisect_left：v 恰等于 bins[k] 落桶 k（含下界不含上界）
    # bins=[100,200,300]; v=100→0, v=200→1, v=300→2, v=350→3
    for v, b in [(100.0, 0), (200.0, 1), (300.0, 2), (350.0, 3), (50.0, 0)]:
        led = _compare(solve, [math.log(1)], None, 1.0, [1.0], v, 1.0, 100.0, 300.0, 4, False)
        assert led["feat_bucket"] == b, (v, led["feat_bucket"])


def check_p_control_before_bucketize(solve):
    # control 乘在 bucketize 之前：150*2=300 → 桶 2（恰等上界）
    led = _compare(solve, [math.log(1)], None, 1.0, [1.0], 150.0, 2.0, 100.0, 300.0, 4, False)
    assert led["feat_bucket"] == 2
    # 1.0 控制：150 → 桶 1
    led2 = _compare(solve, [math.log(1)], None, 1.0, [1.0], 150.0, 1.0, 100.0, 300.0, 4, False)
    assert led2["feat_bucket"] == 1


def check_log_scale_float_endpoint_quirk(solve):
    # log bins: exp(linspace(log 50, log 800, 255))，bins[0] = 49.99... < 50
    # v = 50（标称 min）落桶 1 而非 0 —— 浮点端点 quirk 必须复现
    led = _compare(solve, [math.log(1)], None, 1.0, [1.0], 50.0, 1.0, 50.0, 800.0, 256, True)
    assert led["feat_bucket"] == 1
    # v = 800 → 最后一桶 255
    led2 = _compare(solve, [math.log(1)], None, 1.0, [1.0], 800.0, 1.0, 50.0, 800.0, 256, True)
    assert led2["feat_bucket"] == 255
    led3 = _compare(solve, [math.log(1)], None, 1.0, [1.0], 799.9, 1.0, 50.0, 800.0, 256, True)
    assert led3["feat_bucket"] == 254


def check_linear_scale_exact_endpoint(solve):
    # linear linspace 端点精确：v=min → 桶 0
    led = _compare(solve, [math.log(1)], None, 1.0, [1.0], 50.0, 1.0, 50.0, 800.0, 256, False)
    assert led["feat_bucket"] == 0


def check_empty_sequence(solve):
    led = _compare(solve, [], None, 1.0, [], 0.0, 1.0, 1.0, 2.0, 2, False)
    assert led["duration_rounded"] == [] and led["mel_len"] == 0
    led2 = _compare(solve, [], None, 1.0, [], 0.0, 1.0, 1.0, 2.0, 2, False)
    assert led2["mel_len"] == 0 and led2["feat_bucket"] == 0


def check_invalid_inputs(solve):
    # duration_target 长度必须等于 T
    try:
        solve(H1, [1, 2, 3], 1.0, FEATS, 0.0, 1.0, 1.0, 2.0, 2, False)
        raise SystemError("expected assertion")
    except AssertionError:
        pass
    # n_bins < 2 时 bins 为空 → bucket 0 且不崩溃
    led = solve([math.log(1)], None, 1.0, [1.0], 999.0, 1.0, 1.0, 2.0, 1, False)
    assert led["feat_bucket"] == 0


def check_random_cross_check(solve):
    rng = random.Random(20261101)
    for _ in range(200):
        T = rng.randint(0, 8)
        logd = [rng.uniform(-2.0, 3.0) for _ in range(T)]
        feats = [rng.uniform(-1, 1) for _ in range(T)]
        dc = rng.choice([0.5, 0.8, 1.0, 1.25, 1.5, 2.0])
        use_target = rng.random() < 0.3
        target = [rng.randint(0, 4) for _ in range(T)] if use_target else None
        fmin = rng.uniform(10, 100)
        fmax = fmin + rng.uniform(1, 1000)
        nb = rng.randint(2, 64)
        logs = rng.random() < 0.5
        fv = rng.uniform(0, fmax)
        pc = rng.choice([0.5, 1.0, 1.5])
        _compare(solve, logd, target, dc, feats, fv, pc, fmin, fmax, nb, logs)


import random  # noqa: E402
