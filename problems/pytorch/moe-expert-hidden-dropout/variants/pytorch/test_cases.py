import math

import torch


def _oracle(logits, x, w1, b1, w2, b2, drop_mask, k, p, training):
    """纯 Python：逐行 softmax、按 (-score, index) 选 top-k、fsum 做点积与前向"""
    rows = logits.tolist()
    xs = x.tolist()
    ups = w1.tolist()
    ups_b = b1.tolist()
    downs = w2.tolist()
    downs_b = b2.tolist()
    masks = drop_mask.tolist()

    tokens = len(rows)
    experts = len(rows[0])
    dim = len(xs[0])
    hidden = len(ups[0][0])

    all_idx, all_gates, all_hidden, all_out = [], [], [], []
    for t in range(tokens):
        top = max(rows[t])
        exp = [math.exp(v - top) for v in rows[t]]
        denom = math.fsum(exp)
        scores = [v / denom for v in exp]
        idx = sorted(range(experts), key=lambda e: (-scores[e], e))[:k]
        raw = [scores[e] for e in idx]
        gate_sum = math.fsum(raw)
        gates = [v / gate_sum for v in raw]

        out = [0.0] * dim
        hidden_rows = []
        for j, e in enumerate(idx):
            pre = [math.fsum(xs[t][d] * ups[e][d][h] for d in range(dim)) + ups_b[e][h]
                   for h in range(hidden)]
            h_row = [v if v > 0.0 else 0.0 for v in pre]
            if training and p > 0.0:
                h_row = [v * masks[t][e][h] / (1.0 - p) for h, v in enumerate(h_row)]
            hidden_rows.append(h_row)
            y = [math.fsum(h_row[h] * downs[e][h][d] for h in range(hidden)) + downs_b[e][d]
                 for d in range(dim)]
            for d in range(dim):
                out[d] += gates[j] * y[d]

        all_idx.append(idx)
        all_gates.append(gates)
        all_hidden.append(hidden_rows)
        all_out.append(out)

    return (torch.tensor(all_idx, dtype=torch.int64),
            torch.tensor(all_gates, dtype=torch.float64),
            torch.tensor(all_hidden, dtype=torch.float64),
            torch.tensor(all_out, dtype=torch.float64))


def _verify(solve, args):
    idx, gates, hidden, output = solve(*args)
    tokens, experts = args[0].shape
    k = args[7]
    dim = args[1].shape[1]
    width = args[2].shape[2]
    assert idx.shape == (tokens, k) and idx.dtype == torch.int64
    assert gates.shape == (tokens, k) and gates.dtype == torch.float64
    assert hidden.shape == (tokens, k, width) and hidden.dtype == torch.float64
    assert output.shape == (tokens, dim) and output.dtype == torch.float64
    for value in (gates, hidden, output):
        assert bool(torch.isfinite(value).all())
    assert bool((idx >= 0).all()) and bool((idx < experts).all())
    return idx, gates, hidden, output


def _case(tokens=4, experts=4, dim=3, width=3, k=2, p=0.5, training=True, seed=7,
          contiguous=True):
    gen = torch.Generator().manual_seed(seed)
    logits = torch.randn((tokens, experts), generator=gen, dtype=torch.float64)
    x = torch.randn((tokens, dim), generator=gen, dtype=torch.float64)
    w1 = torch.randn((experts, dim, width), generator=gen, dtype=torch.float64) * 0.7
    b1 = torch.randn((experts, width), generator=gen, dtype=torch.float64) * 0.2
    w2 = torch.randn((experts, width, dim), generator=gen, dtype=torch.float64) * 0.7
    b2 = torch.randn((experts, dim), generator=gen, dtype=torch.float64) * 0.2
    mask = (torch.rand((tokens, experts, width), generator=gen, dtype=torch.float64) > p).to(torch.float64)
    if not contiguous:
        x = torch.randn((tokens, dim * 2), generator=gen, dtype=torch.float64)[:, ::2]
        w1_big = torch.randn((experts, dim, width * 2), generator=gen, dtype=torch.float64) * 0.7
        w1 = w1_big[:, :, ::2]
        mask_big = (torch.rand((tokens, experts, width * 2), generator=gen,
                               dtype=torch.float64) > p).to(torch.float64)
        mask = mask_big[:, :, ::2]
    return (logits, x, w1, b1, w2, b2, mask, k, p, training)


def check_example(solve):
    logits = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float64)
    x = torch.tensor([[2.0, -1.0]], dtype=torch.float64)
    w1 = torch.tensor([
        [[1.0, 0.0], [0.0, 1.0]],
        [[0.0, 1.0], [1.0, 0.0]],
        [[1.0, 1.0], [1.0, 1.0]],
    ], dtype=torch.float64)
    b1 = torch.zeros((3, 2), dtype=torch.float64)
    w2 = torch.eye(2, dtype=torch.float64).repeat(3, 1, 1)
    b2 = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=torch.float64)
    mask = torch.zeros((1, 3, 2), dtype=torch.float64)
    mask[0, 0] = torch.tensor([1.0, 0.0])
    mask[0, 1] = torch.tensor([0.0, 1.0])

    idx, gates, hidden, output = _verify(solve, (logits, x, w1, b1, w2, b2, mask, 2, 0.5, True))
    # softmax(0,0,-1) 的两个 0 并列，选 e0,e1；重归一化后各 0.5
    assert idx.tolist() == [[0, 1]]
    torch.testing.assert_close(gates, torch.tensor([[0.5, 0.5]], dtype=torch.float64))
    # relu 后分别是 [2,0] 与 [0,2]，inverted dropout 乘 2 再按 mask 清零
    torch.testing.assert_close(hidden, torch.tensor([[[4.0, 0.0], [0.0, 4.0]]], dtype=torch.float64))
    torch.testing.assert_close(output, torch.tensor([[2.5, 2.5]], dtype=torch.float64))

    idx, gates, hidden, output = _verify(solve, (logits, x, w1, b1, w2, b2, mask, 2, 0.5, False))
    assert idx.tolist() == [[0, 1]]
    torch.testing.assert_close(gates, torch.tensor([[0.5, 0.5]], dtype=torch.float64))
    torch.testing.assert_close(hidden, torch.tensor([[[2.0, 0.0], [0.0, 2.0]]], dtype=torch.float64))
    torch.testing.assert_close(output, torch.tensor([[1.5, 1.5]], dtype=torch.float64))


check_example.example = True


def check_matches_oracle(solve):
    configs = [
        (1, 1, 1, 1, 1, 0.0, True),
        (2, 3, 2, 2, 2, 0.25, True),
        (3, 4, 3, 4, 1, 0.5, True),
        (4, 5, 2, 3, 3, 0.75, True),
        (5, 4, 4, 2, 4, 0.5, True),
        (3, 2, 3, 3, 2, 0.9, True),
        (4, 4, 3, 3, 2, 0.5, False),
        (2, 6, 2, 2, 5, 0.6, False),
        (6, 3, 2, 5, 2, 0.3, True),
        (3, 3, 3, 3, 2, 0.5, True),
    ]
    for i, (tokens, experts, dim, width, k, p, training) in enumerate(configs):
        args = _case(tokens, experts, dim, width, k, p, training, seed=11 + i,
                     contiguous=(i % 3 != 2))
        actual = _verify(solve, args)
        expected = _oracle(*args)
        for got, want in zip(actual, expected):
            if got.dtype == torch.int64:
                assert got.tolist() == want.tolist()
            else:
                torch.testing.assert_close(got, want, rtol=1e-9, atol=1e-12)


def check_inference_asymmetry(solve):
    logits, x, w1, b1, w2, b2, mask, k, p, _ = _case(4, 4, 3, 3, 2, 0.5, True, seed=21)
    base = _verify(solve, (logits, x, w1, b1, w2, b2, mask, k, 0.0, False))
    for rate in (0.0, 0.3, 0.9):
        got = _verify(solve, (logits, x, w1, b1, w2, b2, mask, k, rate, False))
        for a, b in zip(got, base):
            if a.dtype == torch.int64:
                assert a.tolist() == b.tolist()
            else:
                torch.testing.assert_close(a, b, rtol=0.0, atol=0.0)
    # 训练态 p=0 也必须退化成恒等，mask 全 0 也不许丢
    zeros = torch.zeros_like(mask)
    zero_rate = _verify(solve, (logits, x, w1, b1, w2, b2, zeros, k, 0.0, True))
    for a, b in zip(zero_rate, base):
        if a.dtype == torch.int64:
            assert a.tolist() == b.tolist()
        else:
            torch.testing.assert_close(a, b, rtol=0.0, atol=0.0)


def check_dropout_scaling(solve):
    logits, x, w1, b1, w2, b2, mask, k, p, _ = _case(3, 4, 3, 4, 2, 0.5, True, seed=31)
    base = _verify(solve, (logits, x, w1, b1, w2, b2, mask, k, 0.0, False))
    ones = torch.ones_like(mask)
    dropped = _verify(solve, (logits, x, w1, b1, w2, b2, ones, k, 0.5, True))
    torch.testing.assert_close(dropped[2], base[2] * 2.0, rtol=1e-12, atol=1e-12)

    zeros = torch.zeros_like(mask)
    killed = _verify(solve, (logits, x, w1, b1, w2, b2, zeros, k, 0.5, True))
    torch.testing.assert_close(killed[2], torch.zeros_like(killed[2]))
    # 隐藏激活全灭时，专家输出只剩 bias，再按门控加权
    expected = torch.einsum("tk,tkd->td", killed[1], b2[killed[0]])
    torch.testing.assert_close(killed[3], expected, rtol=1e-12, atol=1e-12)


def check_gates(solve):
    logits, x, w1, b1, w2, b2, mask, _, p, _ = _case(4, 5, 2, 2, 2, 0.5, True, seed=41)
    experts = logits.shape[1]
    idx, gates, _, _ = _verify(solve, (logits, x, w1, b1, w2, b2, mask, experts, p, True))
    # k=E 时 gate 就是 softmax 按分数降序的排列
    expected = torch.sort(torch.softmax(logits, dim=-1), dim=-1, descending=True).values
    torch.testing.assert_close(torch.sort(gates, dim=-1, descending=True).values,
                               expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(gates.sum(dim=1), torch.ones(4, dtype=torch.float64))

    idx1, gates1, _, _ = _verify(solve, (logits, x, w1, b1, w2, b2, mask, 1, p, True))
    torch.testing.assert_close(gates1, torch.ones((4, 1), dtype=torch.float64))
    assert idx1.shape == (4, 1)

    tied = torch.tensor([[1.0, 1.0, 0.0, -1.0, 0.0]], dtype=torch.float64)
    x1 = torch.zeros((1, 2), dtype=torch.float64)
    idx2, _, _, _ = _verify(solve, (tied, x1, w1, b1, w2, b2, mask[:1], 3, 0.0, False))
    assert idx2.tolist() == [[0, 1, 2]]


def check_zero_width_and_single_token(solve):
    logits = torch.tensor([[2.0, -3.0]], dtype=torch.float64)
    x = torch.tensor([[1.5]], dtype=torch.float64)
    w1 = torch.tensor([[[1.0]], [[-1.0]]], dtype=torch.float64)
    b1 = torch.tensor([[0.5], [-0.5]], dtype=torch.float64)
    w2 = torch.tensor([[[2.0]], [[3.0]]], dtype=torch.float64)
    b2 = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    mask = torch.tensor([[[0.0], [1.0]]], dtype=torch.float64)
    actual = _verify(solve, (logits, x, w1, b1, w2, b2, mask, 1, 0.5, True))
    expected = _oracle(logits, x, w1, b1, w2, b2, mask, 1, 0.5, True)
    for got, want in zip(actual, expected):
        if got.dtype == torch.int64:
            assert got.tolist() == want.tolist()
        else:
            torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-12)


def check_invalid_inputs(solve):
    base = _case(2, 3, 2, 2, 2, 0.5, True, seed=51)

    def expect_error(*args):
        try:
            solve(*args)
        except ValueError:
            return
        raise AssertionError("expected ValueError")

    for position, bad in (
        (0, base[0].to(torch.float32)),
        (1, base[1].to(torch.float32)),
        (2, base[2].to(torch.float32)),
        (6, base[6].to(torch.int64)),
    ):
        args = list(base)
        args[position] = bad
        expect_error(*args)

    args = list(base)
    args[6] = torch.full_like(base[6], 0.5)
    expect_error(*args)

    for k in (0, 4, True, 2.0):
        args = list(base)
        args[7] = k
        expect_error(*args)

    for p in (-0.1, 1.0, 1.5, True, float("nan"), float("inf")):
        args = list(base)
        args[8] = p
        expect_error(*args)

    for training in (1, 0, "yes", None):
        args = list(base)
        args[9] = training
        expect_error(*args)

    args = list(base)
    args[0] = base[0][:1]
    expect_error(*args)

    args = list(base)
    args[2] = base[2][:, :, :1]
    expect_error(*args)

    args = list(base)
    args[4] = base[4][:2]
    expect_error(*args)

    nan_logits = base[0].clone()
    nan_logits[0, 0] = float("nan")
    args = list(base)
    args[0] = nan_logits
    expect_error(*args)
