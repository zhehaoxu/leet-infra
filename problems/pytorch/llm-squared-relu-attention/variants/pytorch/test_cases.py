import math

import numpy as np
import torch


def _close(a, b, tol=1e-6):
    a = torch.as_tensor(a, dtype=torch.float64)
    b = torch.as_tensor(b, dtype=torch.float64)
    return torch.allclose(a, b, rtol=tol, atol=tol * max(1.0, float(a.abs().max())))


def _oracle(query, key, value, mask=None, qk_norm=False, eps=1e-6):
    """numpy 重算：relu² 之后按行归一，分母下界 eps，不共享任何向量化代码。"""
    q = np.array(query, dtype=np.float64)
    k = np.array(key, dtype=np.float64)
    v = np.array(value, dtype=np.float64)
    if qk_norm:
        q = q / np.sqrt((q**2).mean(axis=-1, keepdims=True) + eps)
        k = k / np.sqrt((k**2).mean(axis=-1, keepdims=True) + eps)
    logits = (q @ k.swapaxes(-1, -2)) / math.sqrt(q.shape[-1])
    attn = np.maximum(logits, 0.0) ** 2
    if mask is not None:
        keep = np.broadcast_to(np.asarray(mask, dtype=bool), attn.shape)
        attn = np.where(keep, attn, 0.0)
    denom = attn.sum(axis=-1, keepdims=True)
    weights = attn / np.maximum(denom, eps)
    return weights @ v, weights


def _rand(seed, *shape, low=-1.5, high=1.5):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(*shape, generator=g, dtype=torch.float64) * (high - low) + low


def check_example(solve):
    query = torch.tensor([[1.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
    key = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=torch.float64)
    value = torch.tensor([[1.0, 0.0], [0.0, 1.0], [5.0, 5.0]], dtype=torch.float64)
    output, weights = solve(query, key, value)
    expect_w = torch.tensor([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]], dtype=torch.float64)
    expect_o = torch.tensor([[1.0, 0.0], [0.5, 0.5]], dtype=torch.float64)
    assert weights.shape == (2, 3) and output.shape == (2, 2)
    assert weights.dtype == torch.float32 and output.dtype == torch.float32
    assert _close(weights, expect_w, 1e-6), weights
    assert _close(output, expect_o, 1e-6), output


check_example.example = True


def check_activation_zeroing(solve):
    query = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    key = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-2.0, 0.0]], dtype=torch.float64)
    value = torch.eye(3, dtype=torch.float64)
    output, weights = solve(query, key, value)
    # logit 为 0 的位置 relu 后也是 0，必须严格拿 0 权重（softmax 会给 1/3）
    assert weights[0, 0] == 1.0 and weights[0, 1] == 0.0 and weights[0, 2] == 0.0, weights
    assert torch.equal(output, torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32))
    # 一般性质：权重非负；有正 logit 的行和为 1，整行非正的行全 0
    q = _rand(11, 3, 4)
    k = _rand(12, 5, 4)
    v = _rand(13, 5, 2)
    out, w = solve(q, k, v)
    assert bool((w >= 0).all()), w
    positive = ((q @ k.T) / math.sqrt(4)) > 0
    for row in range(3):
        if bool(positive[row].any()):
            assert _close(w[row].sum(), torch.tensor(1.0, dtype=torch.float64), 1e-6), w[row]
        else:
            assert torch.equal(w[row], torch.zeros(5, dtype=torch.float32)), w[row]
    # 输出只由正 logit 的 value 线性组合：把负 logit 的 value 换成任意值都不影响
    rows = [r for r in range(3) if bool(positive[r].any()) and bool((~positive[r]).any())]
    assert rows, "本检查需要一行同时含正负 logit"
    row = rows[0]
    altered = v.clone()
    altered[~positive[row]] = 100.0
    out2, _ = solve(q, k, altered)
    assert _close(out2[row], out[row], 1e-6), (out[row], out2[row])


def check_not_shift_invariant(solve):
    query = torch.tensor([[2.0, 0.0]], dtype=torch.float64)
    key = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]], dtype=torch.float64)
    shifted = key + torch.tensor([2.0, 0.0], dtype=torch.float64)
    raw_scores = (query @ key.T) / math.sqrt(2.0)
    shifted_scores = (query @ shifted.T) / math.sqrt(2.0)
    # 确认这次改动确实是"所有 logit 同加一个常数"
    assert _close(
        shifted_scores - raw_scores,
        torch.full((1, 3), 2.0 * math.sqrt(2.0), dtype=torch.float64),
        1e-9,
    )
    assert _close(
        torch.softmax(shifted_scores, -1), torch.softmax(raw_scores, -1), 1e-9
    ), "softmax 应当平移不变"
    value = torch.eye(3, dtype=torch.float64)
    _, w_raw = solve(query, key, value)
    _, w_shift = solve(query, shifted, value)
    assert not _close(w_raw, w_shift, 1e-3), (w_raw, w_shift)
    # 平移后原本为负的第三个 logit 变成正的，权重从 0 变成正数
    assert w_raw[0, 2] == 0.0 and float(w_shift[0, 2]) > 0.01, (w_raw, w_shift)


def check_squared_growth(solve):
    query = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    key = torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=torch.float64)
    value = torch.eye(3, dtype=torch.float64)
    _, weights = solve(query, key, value)
    # logits 比例 1:2:3，激活是平方 → 权重比例 1:4:9
    assert _close(weights[0, 1] / weights[0, 0], torch.tensor(4.0), 1e-6), weights
    assert _close(weights[0, 2] / weights[0, 0], torch.tensor(9.0), 1e-6), weights
    assert _close(weights, torch.tensor([[1 / 14, 4 / 14, 9 / 14]], dtype=torch.float64), 1e-6)


def check_scale_invariance(solve):
    query = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    key = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [1.0, -1.0]], dtype=torch.float64)
    value = _rand(23, 4, 2)
    out1, w1 = solve(query, key, value)
    assert _close(w1.sum(-1), torch.ones(2, dtype=torch.float64), 1e-6), w1
    # relu² 二次齐次 + 行归一 → 任何正的常数温度都被精确抵消
    for factor in (7.0, 0.05, 1e3):
        out_f, w_f = solve(query * factor, key, value)
        assert _close(w1, w_f, 1e-6), (factor, w1, w_f)
        assert _close(out1, out_f, 1e-6), (factor, out1, out_f)


def check_qk_norm(solve):
    query = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
    key = torch.tensor([[1.0, 1.0], [100.0, 100.0], [0.0, 1.0]], dtype=torch.float64)
    value = torch.eye(3, dtype=torch.float64)
    _, w_plain = solve(query, key, value)
    _, w_norm = solve(query, key, value, qk_norm=True)
    # 未归一化时 100 倍范数的那一行几乎独占全部权重
    assert float(w_plain[0, 1]) > 0.99, w_plain
    # RMSNorm 后方向相同的前两行拿到完全相同的分数与权重
    assert _close(w_norm[0, 0], w_norm[0, 1], 1e-6), w_norm
    assert float(w_norm[0, 0]) < 0.5, w_norm
    assert not _close(w_plain, w_norm, 1e-3)
    # 逐 key 缩放不变性：qk_norm 下每行缩放不改变权重，未归一化则会变
    scales = torch.tensor([[1.0], [0.2], [1.0]], dtype=torch.float64)
    _, w_scaled_norm = solve(query, key * scales, value, qk_norm=True)
    _, w_scaled_plain = solve(query, key * scales, value)
    assert _close(w_norm, w_scaled_norm, 1e-4), (w_norm, w_scaled_norm)
    assert not _close(w_plain, w_scaled_plain, 1e-3)
    # 归一化后 |logit| <= sqrt(hd)：与 q 反向、范数再大的 key 也拿不到权重
    anti_key = torch.tensor([[-1.0, -1.0], [0.0, 0.0]], dtype=torch.float64)
    out, w = solve(query, anti_key, torch.ones(2, 1, dtype=torch.float64), qk_norm=True)
    assert torch.equal(w, torch.zeros(1, 2, dtype=torch.float32)), w
    assert torch.equal(out, torch.zeros(1, 1, dtype=torch.float32)), out


def check_zero_and_tiny_denominator(solve):
    value = torch.ones(2, 1, dtype=torch.float64)
    # q 全零 → 所有 logit 为 0 → 分母为 0 → 权重与输出全 0，且不能出现 NaN
    out, w = solve(
        torch.zeros(1, 2, dtype=torch.float64),
        torch.eye(3, 2, dtype=torch.float64),
        torch.ones(3, 1, dtype=torch.float64),
    )
    assert torch.equal(w, torch.zeros(1, 3, dtype=torch.float32)), w
    assert torch.equal(out, torch.zeros(1, 1, dtype=torch.float32)), out
    # 整行 logit 为负 → 分母 0 → 全 0
    out, w = solve(
        torch.tensor([[-1.0, 0.0]], dtype=torch.float64),
        torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64),
        value,
    )
    assert torch.equal(w, torch.zeros(1, 2, dtype=torch.float32)), w
    # 0 < 分母 < eps：整行被等比缩小，行和 = 分母 / eps < 1
    out, w = solve(
        torch.tensor([[1e-4]], dtype=torch.float64),
        torch.tensor([[1e-4], [-1.0]], dtype=torch.float64),
        value,
    )
    expected = (1e-4 * 1e-4) ** 2 / 1e-6
    assert float(w.sum()) < 1.0 and not bool(torch.isnan(w).any())
    assert abs(float(w[0, 0]) - expected) < 1e-12, (w, expected)
    assert abs(float(out[0, 0]) - expected) < 1e-12


def check_mask(solve):
    query = _rand(31, 2, 3, 4)
    key = _rand(32, 5, 4)
    value = _rand(33, 5, 2)
    mask = torch.tensor([True, False, True, True, False])
    out_b, w_b = solve(query, key, value, mask)
    _, w_full = solve(query, key, value, mask.expand(2, 3, 5))
    assert _close(w_b, w_full, 1e-6)
    assert bool((w_b[..., 1] == 0).all()) and bool((w_b[..., 4] == 0).all())
    # 整行 mask：权重与输出全 0
    full_mask = torch.ones(2, 3, 5, dtype=torch.bool)
    full_mask[:, 0, :] = False
    _, w_row = solve(query, key, value, full_mask)
    assert torch.equal(w_row[:, 0], torch.zeros(2, 5, dtype=torch.float32)), w_row[:, 0]
    assert _close(w_row[:, 1].sum(-1), torch.ones(2, dtype=torch.float64), 1e-6)


def check_broadcast_batch(solve):
    query = _rand(41, 2, 3, 4, 4)
    key = _rand(42, 2, 1, 6, 4)
    value = _rand(43, 2, 1, 6, 3)
    output, weights = solve(query, key, value)
    assert weights.shape == (2, 3, 4, 6), weights.shape
    assert output.shape == (2, 3, 4, 3), output.shape
    ref_out, ref_w = _oracle(query.numpy(), key.numpy(), value.numpy())
    assert _close(weights, ref_w, 1e-6)
    assert _close(output, ref_out, 1e-6)
    # 单头对多头广播：q (1,1,L,hd) 与 k (2,2,S,hd)
    q2 = _rand(44, 1, 1, 3, 4)
    k2 = _rand(45, 2, 2, 5, 4)
    v2 = _rand(46, 2, 2, 5, 2)
    o2, w2 = solve(q2, k2, v2, qk_norm=True)
    assert w2.shape == (2, 2, 3, 5) and o2.shape == (2, 2, 3, 2)
    ref_o2, ref_w2 = _oracle(q2.numpy(), k2.numpy(), v2.numpy(), qk_norm=True)
    assert _close(w2, ref_w2, 1e-6) and _close(o2, ref_o2, 1e-6)


def check_against_oracle(solve):
    cases = [
        (_rand(51, 1, 1, 1), _rand(52, 1, 1), _rand(53, 1, 2), None, False),
        (_rand(54, 4, 3), _rand(55, 7, 3), _rand(56, 7, 2), None, False),
        (_rand(57, 2, 5, 4), _rand(58, 9, 4), _rand(59, 9, 3), None, True),
        (
            _rand(60, 3, 6),
            _rand(61, 5, 6),
            _rand(62, 5, 2),
            torch.tensor([True, True, False, True, False]),
            True,
        ),
        (
            _rand(63, 2, 4),
            _rand(64, 5, 4),
            _rand(65, 5, 1),
            torch.tensor(
                [[True, False, False, True, True], [False, True, True, True, False]]
            ),
            False,
        ),
        (_rand(66, 1, 3) * 1e-4, _rand(67, 3, 3) * 1e-4, _rand(68, 3, 2), None, False),
    ]
    for query, key, value, mask, norm in cases:
        output, weights = solve(query, key, value, mask, qk_norm=norm)
        ref_out, ref_w = _oracle(
            query.numpy(),
            key.numpy(),
            value.numpy(),
            None if mask is None else mask.numpy(),
            qk_norm=norm,
        )
        assert _close(weights, ref_w, 1e-6), (query.shape, norm, weights, ref_w)
        assert _close(output, ref_out, 1e-6), (query.shape, norm, output, ref_out)
    # float32 输入同样可跑，输出仍是 float32
    o32, w32 = solve(
        cases[1][0].to(torch.float32), cases[1][1].to(torch.float32), cases[1][2].to(torch.float32)
    )
    assert o32.dtype == torch.float32 and w32.dtype == torch.float32
    assert _close(w32, _oracle(cases[1][0].numpy(), cases[1][1].numpy(), cases[1][2].numpy())[1], 1e-5)


def check_invalid_inputs(solve):
    q = torch.zeros(2, 3, 4, dtype=torch.float64)
    k = torch.zeros(5, 4, dtype=torch.float64)
    v = torch.zeros(5, 2, dtype=torch.float64)
    bad_calls = [
        ("q-not-tensor", lambda: solve([[0.0] * 4], k, v)),
        ("q-int", lambda: solve(q.to(torch.int64), k, v)),
        ("q-1d", lambda: solve(torch.zeros(4, dtype=torch.float64), k, v)),
        ("q-nan", lambda: solve(torch.full((2, 3, 4), float("nan"), dtype=torch.float64), k, v)),
        ("k-inf", lambda: solve(q, torch.full((5, 4), float("inf"), dtype=torch.float64), v)),
        ("v-not-tensor", lambda: solve(q, k, [[0.0, 0.0]] * 5)),
        ("v-bool", lambda: solve(q, k, torch.zeros(5, 2, dtype=torch.bool))),
        ("head-dim-mismatch", lambda: solve(q, torch.zeros(5, 5, dtype=torch.float64), v)),
        ("kv-length-mismatch", lambda: solve(q, k, torch.zeros(4, 2, dtype=torch.float64))),
        ("k-empty-head", lambda: solve(q, torch.zeros(5, 0, dtype=torch.float64), torch.zeros(5, 2, dtype=torch.float64))),
        (
            "batch-not-broadcastable",
            lambda: solve(
                torch.zeros(2, 3, 4, 4, dtype=torch.float64),
                torch.zeros(5, 1, 5, 4, dtype=torch.float64),
                torch.zeros(5, 1, 5, 2, dtype=torch.float64),
            ),
        ),
        ("mask-not-tensor", lambda: solve(q, k, v, [[True] * 5] * 2)),
        ("mask-float", lambda: solve(q, k, v, torch.ones(2, 5))),
        ("mask-not-broadcastable", lambda: solve(q, k, v, torch.ones(3, dtype=torch.bool))),
        ("qk-norm-int", lambda: solve(q, k, v, None, 1)),
        ("qk-norm-string", lambda: solve(q, k, v, None, "yes")),
        ("eps-bool", lambda: solve(q, k, v, None, False, True)),
        ("eps-zero", lambda: solve(q, k, v, None, False, 0.0)),
        ("eps-negative", lambda: solve(q, k, v, None, False, -1e-6)),
        ("eps-nan", lambda: solve(q, k, v, None, False, float("nan"))),
        ("eps-string", lambda: solve(q, k, v, None, False, "1e-6")),
    ]
    for name, call in bad_calls:
        try:
            call()
        except ValueError:
            continue
        raise AssertionError(f"{name} should raise ValueError")
