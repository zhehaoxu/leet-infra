import math

import torch


def _sig(z):
    """数值稳定的标量 sigmoid：负数走 exp(z)/(1+exp(z))，避免 math.exp(800) 溢出"""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _oracle(logits, bias, k, tau):
    """纯 Python 标量实现：限幅、sigmoid、按 (-sel, index) 排序、用不带 bias 的 s 归一化"""
    rows = logits.tolist()
    b = None if bias is None else bias.tolist()
    e = len(rows[0])
    weights, indices, scores = [], [], []
    for row in rows:
        clamped = [tau * math.tanh(v / tau) for v in row] if tau > 0 else list(row)
        s = [_sig(v) for v in clamped]
        sel = s if b is None else [x + y for x, y in zip(s, b)]
        picked = sorted(range(e), key=lambda i: (-sel[i], i))[:k]
        values = [s[i] for i in picked]
        total = math.fsum(values)
        weights.append([v / total for v in values] if total > 0 else [1.0 / k] * k)
        indices.append(picked)
        scores.append(s)
    return (torch.tensor(weights, dtype=torch.float64),
            torch.tensor(indices, dtype=torch.int64),
            torch.tensor(scores, dtype=torch.float64))


def _case(t=4, e=4, k=2, seed=11, tau=1.0, with_bias=True, scale=3.0):
    rng = torch.Generator().manual_seed(seed)
    logits = torch.rand((t, e), generator=rng, dtype=torch.float64) * (2 * scale) - scale
    bias = None
    if with_bias:
        bias = torch.rand((e,), generator=rng, dtype=torch.float64) * 0.6 - 0.3
    return logits, bias, k, tau


def _verify(solve, args):
    before = {id(v): v.clone() for v in args[:2] if isinstance(v, torch.Tensor)}
    expected = _oracle(*args)
    result = solve(*args)
    assert type(result) is tuple and len(result) == 3
    for actual, target in zip(result, expected):
        assert isinstance(actual, torch.Tensor) and actual.device.type == "cpu"
        assert actual.dtype == target.dtype and actual.shape == target.shape, (actual.shape, target.shape)
        if target.dtype == torch.int64:
            assert torch.equal(actual, target)
        else:
            torch.testing.assert_close(actual, target, rtol=2e-12, atol=2e-12)
    for key, snapshot in before.items():
        for v in args[:2]:
            if isinstance(v, torch.Tensor) and id(v) == key:
                assert torch.equal(v, snapshot)
    return result


def check_example(solve):
    logits = torch.tensor([[2.0, 0.0, -1.0], [1.0, 1.0, 0.5]], dtype=torch.float64)
    weights, indices, scores = _verify(solve, (logits, None, 2, 1.0))
    # tanh(2)=0.9640275800758169 -> sigmoid=0.7239274686640463；tanh(0)=0 -> 0.5；tanh(-1)=-0.7615941559557649 -> 0.3183002578054738
    torch.testing.assert_close(scores[0], torch.tensor(
        [0.7239274686640463, 0.5, 0.3183002578054738], dtype=torch.float64), rtol=0, atol=1e-15)
    torch.testing.assert_close(weights[0], torch.tensor(
        [0.5914790599922027, 0.4085209400077972], dtype=torch.float64), rtol=0, atol=1e-15)
    # 第二行两个 1.0 限幅后完全相同 -> 严格并列，按编号取 0、1，权重各 1/2
    assert indices[1].tolist() == [0, 1]
    torch.testing.assert_close(weights[1], torch.tensor([0.5, 0.5], dtype=torch.float64), rtol=0, atol=0)
    assert float(scores[1][0]) == float(scores[1][1])
    # 加上 bias 之后选择会换人，但权重仍由不带 bias 的 s 归一化
    bias = torch.tensor([0.1, -0.2, 0.0], dtype=torch.float64)
    w2, i2, _ = _verify(solve, (logits, bias, 2, 1.0))
    assert i2[0].tolist() == [0, 2]
    torch.testing.assert_close(w2[0], torch.tensor(
        [0.7239274686640463 / (0.7239274686640463 + 0.3183002578054738),
         0.3183002578054738 / (0.7239274686640463 + 0.3183002578054738)], dtype=torch.float64),
        rtol=0, atol=1e-15)


check_example.example = True


def check_matches_oracle(solve):
    for seed in range(1, 9):
        for tau in (0.0, 0.5, 1.0, 8.0):
            for with_bias in (True, False):
                _verify(solve, _case(seed=seed, tau=tau, with_bias=with_bias,
                                     k=1 + seed % 4, scale=[1.0, 4.0, 20.0][seed % 3]))


def check_saturation_ties(solve):
    # tanh(20) 与 tanh(21) 在 float64 里都恰好是 1.0，限幅把两者压成严格并列
    logits = torch.tensor([[20.0, 21.0, -5.0]], dtype=torch.float64)
    weights, indices, scores = _verify(solve, (logits, None, 2, 1.0))
    assert float(scores[0][0]) == float(scores[0][1]) == _sig(1.0)
    assert indices[0].tolist() == [0, 1]
    torch.testing.assert_close(weights[0], torch.tensor([0.5, 0.5], dtype=torch.float64), rtol=0, atol=0)
    # tau 极大时限幅近似恒等，排序与关闭限幅完全一致
    big = _case(t=5, e=6, k=3, seed=21, tau=0.0, with_bias=False, scale=6.0)
    off_w, off_i, _ = solve(big[0], None, 3, 0.0)
    _w, _i, _s = solve(big[0], None, 3, 1e6)
    assert torch.equal(_i, off_i)
    torch.testing.assert_close(_w, off_w, rtol=1e-9, atol=1e-12)
    # tau 极小时所有分数挤在 0.5 附近，bias 反过来决定选择
    tiny = torch.tensor([[100.0, -100.0, 0.0]], dtype=torch.float64)
    bias = torch.tensor([0.0, 0.0, 0.1], dtype=torch.float64)
    _w, i_t, s_t = solve(tiny, bias, 2, 1e-3)
    assert i_t[0].tolist() == [2, 0]
    assert float(s_t[0][2]) == 0.5 and abs(float(s_t[0][0]) - 0.5) < 3e-4


def check_bias_stays_out_of_weights(solve):
    logits = torch.tensor([[0.0, -2.0, 2.0]], dtype=torch.float64)
    bias = torch.tensor([0.0, 0.9, 0.0], dtype=torch.float64)
    weights, indices, _ = _verify(solve, (logits, bias, 2, 0.0))
    assert indices[0].tolist() == [1, 2]
    # s=[0.5, 0.11920292202211755, 0.8807970779778823]；bias 把专家 1 抬进选择，但权重按 s 归一化
    torch.testing.assert_close(weights[0], torch.tensor(
        [0.11920292202211755, 0.8807970779778823], dtype=torch.float64), rtol=0, atol=1e-15)
    # 给所有专家加同一个常数不改变排序，也不改变权重
    case = _case(t=3, e=5, k=2, seed=5, tau=2.0, with_bias=False)
    _w0, i0, _s0 = solve(case[0], None, 2, 2.0)
    _w1, i1, _s1 = solve(case[0], torch.full((5,), 0.37, dtype=torch.float64), 2, 2.0)
    assert torch.equal(i0, i1)
    torch.testing.assert_close(_w0, _w1, rtol=0, atol=1e-15)


def check_edges_and_underflow(solve):
    # k=E：每个专家都选，权重就是整行 sigmoid 的归一化
    case = _case(t=3, e=4, k=4, seed=7, tau=1.5)
    weights, indices, scores = _verify(solve, case)
    for row in range(3):
        assert sorted(indices[row].tolist()) == [0, 1, 2, 3]
        assert abs(float(weights[row].sum()) - 1.0) <= 1e-15
        assert bool((scores[row] > 0).all()) and bool((scores[row] < 1).all())
    # 全零 logits：所有分数并列 0.5，按编号取前 k，权重均匀
    z = torch.zeros(2, 5, dtype=torch.float64)
    weights, indices, scores = _verify(solve, (z, None, 3, 1.0))
    torch.testing.assert_close(weights[1], torch.full((3,), 1 / 3, dtype=torch.float64), rtol=0, atol=1e-15)
    assert indices[1].tolist() == [0, 1, 2]
    assert bool((scores == 0.5).all())
    # sigmoid 下溢：选中项之和为 0，按契约给均匀权重，分数仍如实为 0
    u = torch.tensor([[-800.0, -900.0, 5.0]], dtype=torch.float64)
    weights, indices, scores = _verify(solve, (u, None, 2, 0.0))
    assert indices[0].tolist() == [2, 0]
    assert float(scores[0][0]) == 0.0 and float(scores[0][1]) == 0.0
    torch.testing.assert_close(weights[0], torch.tensor([1.0, 0.0], dtype=torch.float64), rtol=0, atol=0)
    all_zero = torch.tensor([[-800.0, -900.0]], dtype=torch.float64)
    weights, indices, _ = _verify(solve, (all_zero, None, 2, 0.0))
    assert indices[0].tolist() == [0, 1]
    torch.testing.assert_close(weights[0], torch.tensor([0.5, 0.5], dtype=torch.float64), rtol=0, atol=0)
    # 非连续输入也要能算
    nc = torch.rand(4, 3, generator=torch.Generator().manual_seed(3), dtype=torch.float64) * 6 - 3
    _verify(solve, (nc.t().contiguous().t(), None, 2, 1.0))


def check_invalid(solve):
    logits = _case(t=3, e=5, k=2, seed=9, with_bias=False)[0]
    for bad_k in (0, -1, 6, 2.0, True, None, [2]):
        try:
            solve(logits, None, bad_k, 1.0)
        except ValueError:
            pass
        else:
            raise AssertionError(f"k={bad_k!r} should raise ValueError")
    for bad_tau in (-1.0, -0.5, float('nan'), float('inf'), True, '1.0', None, [1.0]):
        try:
            solve(logits, None, 2, bad_tau)
        except ValueError:
            pass
        else:
            raise AssertionError(f"tau={bad_tau!r} should raise ValueError")
    for bad_bias in (torch.zeros(4, dtype=torch.float64), torch.zeros(5, dtype=torch.float32),
                     torch.zeros(1, 5, dtype=torch.float64), torch.zeros(5, dtype=torch.int64),
                     [0.1, 0.2, 0.3, 0.4, 0.5]):
        try:
            solve(logits, bad_bias, 2, 1.0)
        except ValueError:
            pass
        else:
            raise AssertionError("bias should raise ValueError")
    for bad_logits in ([1.0, 2.0], 3, None, torch.rand(5, dtype=torch.float64),
                        torch.rand(2, 2, 2, dtype=torch.float64), torch.rand(2, 3),
                        torch.randint(0, 3, (2, 3), dtype=torch.int64)):
        try:
            solve(bad_logits, None, 1, 1.0)
        except ValueError:
            pass
        else:
            raise AssertionError("logits should raise ValueError")
    # tau=0 与 tau=0.0 都合法
    solve(logits, None, 2, 0)


def check_maximum_shape(solve):
    _verify(solve, _case(t=8, e=8, k=8, seed=31, tau=0.25))
