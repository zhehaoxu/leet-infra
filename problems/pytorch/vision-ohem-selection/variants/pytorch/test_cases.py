"""OHEM 难例选择掩码账（test cases）。"""

import random


def _py(losses, probs, thresh, min_kept, batch_size):
    """独立 oracle：按官方语义逐步重放。"""
    batch_kept = min_kept * batch_size
    if thresh is not None:
        n = len(probs)
        if n > 0:
            sp = sorted(probs)
            k = batch_kept if batch_kept < n else n - 1
            min_threshold = sp[k]
        else:
            min_threshold = 0.0
        threshold = max(min_threshold, thresh)
        mask = []
        for v in probs:
            mask.append(1 if v < threshold else 0)
        return {"mask": mask, "threshold": threshold}
    # loss 分支：降序稳定
    idx = list(range(len(losses)))
    idx.sort(key=lambda i: (-losses[i], i))
    mask = [0] * len(losses)
    for i in idx[:batch_kept]:
        mask[i] = 1
    return {"mask": mask, "threshold": None}


def _close(actual, expect):
    if isinstance(actual, dict):
        for k in expect:
            assert _close(actual[k], expect[k]), f"mismatch at {k}"
        return True
    if isinstance(actual, list):
        assert len(actual) == len(expect)
        return all(_close(a, e) for a, e in zip(actual, expect))
    assert actual == expect, f"{actual!r} != {expect!r}"
    return True


def _compare(solve, losses, probs, thresh, min_kept, batch_size):
    led = solve(losses, probs, thresh, min_kept, batch_size)
    ref = _py(losses, probs, thresh, min_kept, batch_size)
    _close(led, ref)
    return led


def check_anchor_threshold_branch(solve):
    # probs 升序 [0.01..0.05]，min_kept=3, batch=1 → rank3 概率 0.04；
    # t=0.03 → threshold=max(0.04,0.03)=0.04 → 保留 <0.04 的前 3 个
    probs = [0.04, 0.01, 0.05, 0.02, 0.03]
    led = _compare(solve, None, probs, 0.03, 3, 1)
    assert led["mask"] == [0, 1, 0, 1, 1]
    assert led["threshold"] == 0.04


def check_threshold_dominates(solve):
    # min_kept=3 → rank3 兜底 0.4 > t=0.25 → threshold=0.4（rank 兜底胜出）
    probs = [0.1, 0.2, 0.3, 0.4, 0.5]
    led = _compare(solve, None, probs, 0.25, 3, 1)
    assert led["threshold"] == 0.4
    assert led["mask"] == [1, 1, 1, 0, 0]  # <0.4 严格小于


def check_min_kept_backstop(solve):
    # t=0：threshold=rank 兜底值，至少保留 batch_kept 个
    probs = [0.9, 0.1, 0.5, 0.7, 0.3]
    led = _compare(solve, None, probs, 0.0, 2, 1)
    # 升序 [0.1,0.3,0.5,0.7,0.9]，rank2 = 0.5 → threshold 0.5 → 严格小于：0.1、0.3
    assert led["mask"] == [0, 1, 0, 0, 1]
    assert led["threshold"] == 0.5


def check_batch_scaling(solve):
    # batch_kept = min_kept * batch_size：3*2=6，n=8
    probs = [0.08, 0.02, 0.06, 0.04, 0.10, 0.01, 0.07, 0.03]
    led = _compare(solve, None, probs, 0.0, 3, 2)
    sp = sorted(probs)
    want_thr = sp[6]  # min(6, 7) = 6
    assert led["threshold"] == want_thr
    assert sum(led["mask"]) == sum(1 for v in probs if v < want_thr)


def check_loss_branch_anchor(solve):
    # loss 降序 top 3，并列按稳定排序取靠前
    losses = [1.0, 3.0, 2.0, 3.0, 0.5]
    led = _compare(solve, losses, None, None, 3, 1)
    assert led["mask"] == [0, 1, 1, 1, 0]  # 3.0(idx1), 3.0(idx3), 2.0(idx2)
    assert led["threshold"] is None


def check_loss_branch_tie_stability(solve):
    # 全并列：稳定排序取前 batch_kept 个下标
    losses = [1.0] * 5
    led = _compare(solve, losses, None, None, 2, 1)
    assert led["mask"] == [1, 1, 0, 0, 0]


def check_loss_branch_quota_exceeds(solve):
    # 配额 > n：全保留
    losses = [0.1, 0.2]
    led = _compare(solve, losses, None, None, 5, 1)
    assert led["mask"] == [1, 1]


def check_oracle_random(solve):
    rng = random.Random(676)
    for _ in range(200):
        n = rng.randint(1, 30)
        batch = rng.randint(1, 4)
        min_kept = rng.randint(1, 10)
        if rng.random() < 0.5:
            probs = [rng.random() for _ in range(n)]
            thresh = rng.choice([0.0, 0.1, 0.3, 0.5, rng.random()])
            _compare(solve, None, probs, thresh, min_kept, batch)
        else:
            losses = [rng.random() * 5 for _ in range(n)]
            _compare(solve, losses, None, None, min_kept, batch)


def check_invalid_inputs(solve):
    for args in [
        (None, [0.5], 0.5, 0, 1),        # min_kept < 1
        (None, [0.5], 0.5, 1, 0),        # batch_size < 1
        (None, [0.5], -0.1, 1, 1),       # thresh < 0
        (None, [1.5], 0.5, 1, 1),        # prob 越界
        ([0.5], [0.5], 0.5, 1, 1),       # 两分支同时给
        ([None], None, None, 1, 1),      # loss 非有限数
        ([-0.1], None, None, 1, 1),      # 负 loss
    ]:
        try:
            solve(*args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError: {args}")
