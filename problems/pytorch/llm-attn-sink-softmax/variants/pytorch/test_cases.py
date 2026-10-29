import math

import numpy as np
import torch


def _close(a, b, tol=1e-6):
    a = torch.as_tensor(a, dtype=torch.float64)
    b = torch.as_tensor(b, dtype=torch.float64)
    return torch.allclose(a, b, rtol=tol, atol=tol * max(1.0, float(a.abs().max())))


def _oracle(scores, mask=None, sink_logit=0.0):
    """逐行标量实现：被屏蔽的位置连 exp 都不算，与向量化解法互不共享代码。"""
    arr = np.asarray(scores, dtype=np.float64)
    flat = arr.reshape(-1, arr.shape[-1])
    if mask is None:
        keep = np.ones(flat.shape, dtype=bool)
    else:
        keep = np.broadcast_to(np.asarray(mask, dtype=bool), arr.shape).reshape(flat.shape)
    weights = np.zeros(flat.shape, dtype=np.float64)
    sink_weight = np.zeros(flat.shape[0], dtype=np.float64)
    for r in range(flat.shape[0]):
        idx = np.flatnonzero(keep[r])
        ref = float(sink_logit)
        for j in idx:
            if flat[r, j] > ref:
                ref = float(flat[r, j])
        e_sink = math.exp(float(sink_logit) - ref)
        terms = {int(j): math.exp(float(flat[r, j]) - ref) for j in idx}
        denom = sum(terms.values()) + e_sink
        for j, term in terms.items():
            weights[r, j] = term / denom
        sink_weight[r] = e_sink / denom
    return weights.reshape(arr.shape), sink_weight.reshape(arr.shape[:-1])


def _rand(seed, *shape, low=-4.0, high=4.0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(*shape, generator=g, dtype=torch.float64) * (high - low) + low


def _rand_mask(seed, *shape, keep_prob=0.6):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(*shape, generator=g, dtype=torch.float64) < keep_prob


def check_example(solve):
    scores = torch.tensor(
        [[0.0, 0.6931471805599453], [0.0, -3.0], [-1.0, -2.0]], dtype=torch.float64
    )
    mask = torch.tensor([[True, True], [True, False], [False, False]])
    weights, sink_weight = solve(scores, mask, 0.0)
    expect_w = torch.tensor(
        [[0.25, 0.5], [0.5, 0.0], [0.0, 0.0]], dtype=torch.float64
    )
    expect_s = torch.tensor([0.25, 0.5, 1.0], dtype=torch.float64)
    assert weights.shape == scores.shape, weights.shape
    assert sink_weight.shape == (3,), sink_weight.shape
    assert weights.dtype == torch.float32 and sink_weight.dtype == torch.float32
    assert _close(weights, expect_w, 1e-6), weights
    assert _close(sink_weight, expect_s, 1e-6), sink_weight


check_example.example = True


def check_row_sum_deficit(solve):
    scores = _rand(11, 3, 7)
    mask = _rand_mask(12, 3, 7)
    weights, sink_weight = solve(scores, mask, 0.7)
    total = weights.sum(dim=-1) + sink_weight
    assert _close(total, torch.ones(3, dtype=torch.float64), 1e-6), total
    visible = weights.sum(dim=-1)
    assert bool((visible < 1.0).all()), visible
    assert bool((sink_weight > 0.0).all()), sink_weight
    # 被屏蔽位置的权重必须严格是 0，而不是"很小的数"
    assert torch.equal(weights[~mask], torch.zeros(int((~mask).sum()), dtype=weights.dtype))


def check_sink_ratio_invariance(solve):
    scores = torch.tensor([[0.3, -1.2, 2.5]], dtype=torch.float64)
    w0, s0 = solve(scores, None, 0.0)
    w1, s1 = solve(scores, None, 1.5)
    w2, s2 = solve(scores, None, 3.0)
    assert float(s0) < float(s1) < float(s2), (s0, s1, s2)
    assert bool((w0 > w1).all()) and bool((w1 > w2).all()), (w0, w1, w2)
    # 行内比值只由 scores 决定，sink_logit 只是整体缩放
    assert _close(w0[0, 0] / w0[0, 2], math.exp(0.3 - 2.5), 1e-6)
    assert _close(w1[0, 0] / w1[0, 2], float(w0[0, 0] / w0[0, 2]), 1e-6)


def check_all_masked_rows(solve):
    scores = torch.tensor([[1.0, 2.0], [-5.0, 5.0], [0.0, 0.5]], dtype=torch.float64)
    mask = torch.tensor(
        [[False, False], [False, False], [True, False]]
    )
    weights, sink_weight = solve(scores, mask, 0.0)
    assert not bool(torch.isnan(weights).any()) and not bool(torch.isnan(sink_weight).any())
    assert torch.equal(weights[:2], torch.zeros(2, 2, dtype=weights.dtype))
    assert _close(sink_weight[:2], torch.ones(2, dtype=torch.float64), 1e-6)
    # 只剩一个保留位且它等于 sink → 五五开
    assert _close(sink_weight[2], torch.tensor(0.5, dtype=torch.float64), 1e-6)
    assert _close(weights[2, 0], torch.tensor(0.5, dtype=torch.float64), 1e-6)


def check_standard_softmax_limit(solve):
    scores = torch.tensor([[0.0, math.log(2.0)], [2.5, -1.0]], dtype=torch.float64)
    weights, sink_weight = solve(scores, None, -80.0)
    assert _close(weights, torch.softmax(scores, dim=-1), 1e-6), weights
    assert bool((sink_weight < 1e-30).all()), sink_weight
    # 带掩码时退化为 masked softmax（本行至少留一位）
    mask = torch.tensor([[True, False], [True, True]])
    w2, s2 = solve(scores, mask, -80.0)
    ref = torch.softmax(torch.where(mask, scores, torch.full_like(scores, float("-inf"))), -1)
    assert _close(w2, ref, 1e-6), w2
    assert _close(s2, torch.tensor([0.0, 0.0], dtype=torch.float64), 1e-7)


def check_mask_broadcast(solve):
    scores = _rand(21, 2, 3, 5)
    mask1 = torch.tensor([True, False, True, True, False])
    w_b, s_b = solve(scores, mask1, 0.3)
    w_f, s_f = solve(scores, mask1.expand(2, 3, 5), 0.3)
    assert _close(w_b, w_f, 1e-7) and _close(s_b, s_f, 1e-7)
    assert bool((w_b[..., 1] == 0).all()) and bool((w_b[..., 4] == 0).all())


def check_sink_equals_zero_token(solve):
    scores = _rand(31, 2, 6)
    mask = _rand_mask(32, 2, 6, keep_prob=0.7)
    for row in range(2):
        assert bool(mask[row].any()), "本检查要求每行至少保留一位"
    for sink_logit in (-0.5, 0.0, 1.25):
        augmented = torch.cat(
            [
                torch.where(mask, scores, torch.full_like(scores, float("-inf"))),
                torch.full((2, 1), float(sink_logit), dtype=torch.float64),
            ],
            dim=-1,
        )
        ref = torch.softmax(augmented, dim=-1)
        weights, sink_weight = solve(scores, mask, sink_logit)
        assert _close(weights, ref[:, :-1], 1e-6), (sink_logit, weights)
        assert _close(sink_weight, ref[:, -1], 1e-6), (sink_logit, sink_weight)


def check_large_logits_no_overflow(solve):
    scores = torch.tensor([[1000.0, 999.0, -1000.0]], dtype=torch.float64)
    weights, sink_weight = solve(scores, None, 0.0)
    assert bool(torch.isfinite(weights).all()) and bool(torch.isfinite(sink_weight).all())
    denom = 1.0 + math.exp(-1.0)
    expect = torch.tensor(
        [[1.0 / denom, math.exp(-1.0) / denom, 0.0]], dtype=torch.float64
    )
    assert _close(weights, expect, 1e-6), weights
    # sink 远低于最大值时拿不到质量（exp(0-1000) 下溢为 0）
    assert _close(sink_weight, torch.zeros(1, dtype=torch.float64), 1e-6)
    # 反向极端：整行远低于 sink，概率质量几乎全被 sink 吃掉
    w2, s2 = solve(torch.tensor([[-1000.0, -1001.0]], dtype=torch.float64), None, 0.0)
    assert _close(w2, torch.zeros(1, 2, dtype=torch.float64), 1e-6), w2
    assert _close(s2, torch.ones(1, dtype=torch.float64), 1e-6), s2


def check_float64_internal_precision(solve):
    # 大公共偏置 + 极小差：内部若用 float32，(1e5 - 1e-3) 会被量化成 1e5，权重退化成 0.5
    scores = torch.tensor([[1e5, 1e5 - 1e-3]], dtype=torch.float64)
    weights, sink_weight = solve(scores, None, 0.0)
    denom = 1.0 + math.exp(-1e-3)
    expect = torch.tensor(
        [[1.0 / denom, math.exp(-1e-3) / denom]], dtype=torch.float64
    )
    assert _close(weights, expect, 1e-6), weights
    assert _close(sink_weight, torch.zeros(1, dtype=torch.float64), 1e-6)
    # 带掩码时同理：掩码外的巨大 logit 不能污染分母
    mask = torch.tensor([[True, False]])
    w2, s2 = solve(scores, mask, 0.0)
    assert _close(w2, torch.tensor([[1.0, 0.0]], dtype=torch.float64), 1e-6), w2
    assert _close(s2, torch.zeros(1, dtype=torch.float64), 1e-6)


def check_against_oracle(solve):
    cases = [
        (None, _rand(41, 1), 0.0),
        (None, _rand(42, 4), -5.0),
        (None, _rand(43, 1, 6), 3.25),
        (_rand_mask(44, 5), _rand(45, 2, 5), 0.0),
        (_rand_mask(46, 2, 4, keep_prob=0.35), _rand(47, 2, 4), 1.0),
        (_rand_mask(48, 3, 2, 7), _rand(49, 3, 2, 7), -2.5),
        (_rand_mask(50, 2, 2, 3, keep_prob=0.9), _rand(51, 2, 2, 3), 0.0),
    ]
    for mask, scores, sink_logit in cases:
        weights, sink_weight = solve(scores, mask, sink_logit)
        ref_w, ref_s = _oracle(scores.numpy(), None if mask is None else mask.numpy(), sink_logit)
        assert weights.shape == scores.shape, (weights.shape, scores.shape)
        assert sink_weight.shape == scores.shape[:-1], (sink_weight.shape, scores.shape)
        assert _close(weights, ref_w, 1e-6), (scores.shape, sink_logit, weights, ref_w)
        assert _close(sink_weight, ref_s, 1e-6), (scores.shape, sink_logit, sink_weight, ref_s)
    # 整行屏蔽的行 oracle 与解法的约定一致：sink 独占
    scores = torch.tensor([[-2.0, 3.0]], dtype=torch.float64)
    mask = torch.tensor([[False, False]])
    weights, sink_weight = solve(scores, mask, 0.25)
    assert _close(weights, torch.zeros(1, 2, dtype=torch.float64), 1e-7)
    assert _close(sink_weight, torch.ones(1, dtype=torch.float64), 1e-6)


def check_invalid_inputs(solve):
    good = torch.zeros(2, 3, dtype=torch.float64)
    bad_calls = [
        ("scores-not-tensor", lambda: solve([[0.0, 1.0]], None, 0.0)),
        ("scores-int", lambda: solve(torch.zeros(2, 3, dtype=torch.int64), None, 0.0)),
        ("scores-bool", lambda: solve(torch.zeros(2, 3, dtype=torch.bool), None, 0.0)),
        ("scores-scalar", lambda: solve(torch.tensor(1.0, dtype=torch.float64), None, 0.0)),
        ("scores-empty-last", lambda: solve(torch.zeros(2, 0, dtype=torch.float64), None, 0.0)),
        (
            "scores-nan",
            lambda: solve(torch.tensor([[0.0, float("nan")]], dtype=torch.float64), None, 0.0),
        ),
        (
            "scores-posinf",
            lambda: solve(torch.tensor([[0.0, float("inf")]], dtype=torch.float64), None, 0.0),
        ),
        (
            "scores-neginf",
            lambda: solve(torch.tensor([[0.0, float("-inf")]], dtype=torch.float64), None, 0.0),
        ),
        ("mask-not-tensor", lambda: solve(good, [[True, True, True], [True, True, True]], 0.0)),
        ("mask-float", lambda: solve(good, torch.ones(2, 3), 0.0)),
        ("mask-not-broadcastable", lambda: solve(good, torch.ones(4, dtype=torch.bool), 0.0)),
        ("sink-bool", lambda: solve(good, None, True)),
        ("sink-str", lambda: solve(good, None, "0")),
        ("sink-none", lambda: solve(good, None, None)),
        ("sink-nan", lambda: solve(good, None, float("nan"))),
        ("sink-inf", lambda: solve(good, None, float("inf"))),
        ("sink-neginf", lambda: solve(good, None, float("-inf"))),
    ]
    for name, call in bad_calls:
        try:
            call()
        except ValueError:
            continue
        raise AssertionError(f"{name} should raise ValueError")
    # 合法边界：int sink_logit、单元素行、非连续 mask 都应当正常返回
    weights, sink_weight = solve(good, None, 0)
    assert weights.shape == (2, 3) and sink_weight.shape == (2,)
    weights, sink_weight = solve(torch.zeros(1, 1, dtype=torch.float64), None, 0.0)
    assert _close(weights, torch.tensor([[0.5]], dtype=torch.float64), 1e-6)
    assert _close(sink_weight, torch.tensor([0.5], dtype=torch.float64), 1e-6)
