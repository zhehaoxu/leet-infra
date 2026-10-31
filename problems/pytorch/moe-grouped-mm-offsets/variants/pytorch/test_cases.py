import torch


def _oracle(x, weight, offs, grad_output):
    """纯 Python 三重循环：逐组、逐行、逐列做标量乘加，不借用任何张量算子"""
    rows, k_dim = x.shape
    experts = weight.shape[0]
    n_dim = weight.shape[2]
    xs = x.tolist()
    ws = weight.tolist()
    gs = grad_output.tolist()
    ends = offs.tolist()

    out = [[0.0] * n_dim for _ in range(rows)]
    grad_input = [[0.0] * k_dim for _ in range(rows)]
    grad_weight = [[[0.0] * n_dim for _ in range(k_dim)] for _ in range(experts)]
    counts = [0] * experts

    start = 0
    for expert in range(experts):
        end = ends[expert]
        counts[expert] = end - start
        for row in range(start, end):
            # out[row] = x[row] @ weight[expert]
            for col in range(n_dim):
                acc = 0.0
                for t in range(k_dim):
                    acc += xs[row][t] * ws[expert][t][col]
                out[row][col] = acc
            # grad_input[row] = grad[row] @ weight[expert].T
            for t in range(k_dim):
                acc = 0.0
                for col in range(n_dim):
                    acc += gs[row][col] * ws[expert][t][col]
                grad_input[row][t] = acc
        # grad_weight[expert][t][col] = sum over rows in the group of x[row][t] * grad[row][col]
        for t in range(k_dim):
            for col in range(n_dim):
                acc = 0.0
                for row in range(start, end):
                    acc += xs[row][t] * gs[row][col]
                grad_weight[expert][t][col] = acc
        start = end

    return (torch.tensor(out, dtype=torch.float64), torch.tensor(grad_input, dtype=torch.float64),
            torch.tensor(grad_weight, dtype=torch.float64), torch.tensor(counts, dtype=torch.int64))


def _counts_of(assigned, experts, rng):
    """把 assigned 行随机丢进 experts 个盒子，允许任意盒子为空（case 生成，不参与判题逻辑）"""
    if assigned == 0:
        return [0] * experts
    boxes = torch.randint(0, experts, (assigned,), generator=rng)
    return [int((boxes == e).sum()) for e in range(experts)]


def _case(rows=6, experts=3, k=2, n=2, seed=1, assigned=None, quarter=False, floaty=False, counts=None):
    rng = torch.Generator().manual_seed(seed)
    if counts is None:
        if assigned is None:
            assigned = int(torch.randint(0, rows + 1, (1,), generator=rng).item())
        counts = _counts_of(assigned, experts, rng)
    offs, running = [], 0
    for count in counts:
        running += count
        offs.append(running)
    assert running <= rows, "case generator produced offs past the last row"
    if floaty:
        x = torch.randn(rows, k, generator=rng, dtype=torch.float64)
        weight = torch.randn(experts, k, n, generator=rng, dtype=torch.float64)
        grad_output = torch.randn(rows, n, generator=rng, dtype=torch.float64)
    else:
        # 整数（或 0.25 倍数）数据：乘加全程二进制精确，可以和纯 Python 逐位比
        x = torch.randint(-3, 4, (rows, k), generator=rng, dtype=torch.int64).to(torch.float64)
        weight = torch.randint(-3, 4, (experts, k, n), generator=rng, dtype=torch.int64).to(torch.float64)
        grad_output = torch.randint(-3, 4, (rows, n), generator=rng, dtype=torch.int64).to(torch.float64)
        if quarter:
            x = x * 0.25
            weight = weight * 0.25
            grad_output = grad_output * 0.25
    return x, weight, torch.tensor(offs, dtype=torch.int64), grad_output


def _example():
    x = torch.tensor([[1.0, 2.0], [0.0, -1.0], [2.0, 1.0], [1.0, 0.0], [-1.0, 2.0], [3.0, 3.0]],
                     dtype=torch.float64)
    weight = torch.tensor([[[1.0, 0.0], [2.0, 1.0]],
                           [[1.0, 1.0], [1.0, 1.0]],
                           [[0.0, 2.0], [-1.0, 1.0]]], dtype=torch.float64)
    offs = torch.tensor([2, 2, 5], dtype=torch.int64)
    grad_output = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0], [0.0, 2.0], [7.0, 7.0]],
                               dtype=torch.float64)
    return x, weight, offs, grad_output


def _verify(solve, args):
    x, weight, offs, grad_output = args
    before = [value.clone() for value in (x, weight, offs, grad_output)]
    expected = _oracle(x, weight, offs, grad_output)
    result = solve(x, weight, offs, grad_output)
    assert isinstance(result, tuple) and len(result) == 4, f"expected a 4-tuple, got {type(result)}"
    names = ("out", "grad_input", "grad_weight", "counts")
    for name, actual, target in zip(names, result, expected):
        assert actual.dtype == target.dtype and actual.shape == target.shape, (
            name, actual.dtype, actual.shape, target.dtype, target.shape)
        if target.dtype == torch.float64:
            torch.testing.assert_close(actual, target, rtol=2e-11, atol=2e-11)
        else:
            assert torch.equal(actual, target), (name, actual.tolist(), target.tolist())
    for value, snapshot in zip((x, weight, offs, grad_output), before):
        assert torch.equal(value, snapshot)
    return result


def check_example(solve):
    args = _example()
    out, grad_input, grad_weight, counts = _verify(solve, args)
    assert counts.tolist() == [2, 0, 3]
    torch.testing.assert_close(out, torch.tensor(
        [[5.0, 2.0], [-2.0, -1.0], [-1.0, 5.0], [0.0, 2.0], [-2.0, 0.0], [0.0, 0.0]],
        dtype=torch.float64), rtol=0, atol=0)
    torch.testing.assert_close(grad_input, torch.tensor(
        [[1.0, 2.0], [0.0, 1.0], [2.0, 0.0], [0.0, -2.0], [4.0, 2.0], [0.0, 0.0]],
        dtype=torch.float64), rtol=0, atol=0)
    torch.testing.assert_close(grad_weight, torch.tensor(
        [[[1.0, 0.0], [2.0, -1.0]], [[0.0, 0.0], [0.0, 0.0]], [[4.0, 0.0], [1.0, 5.0]]],
        dtype=torch.float64), rtol=0, atol=0)


check_example.example = True


def check_matches_oracle(solve):
    combos = [
        dict(rows=6, experts=3, k=2, n=2, seed=2),
        dict(rows=9, experts=4, k=3, n=2, seed=3, assigned=9),
        dict(rows=7, experts=5, k=2, n=4, seed=5, quarter=True),
        dict(rows=12, experts=6, k=3, n=3, seed=7),
        dict(rows=10, experts=2, k=4, n=5, seed=11, assigned=10, quarter=True),
        dict(rows=8, experts=7, k=2, n=2, seed=13),
        dict(rows=5, experts=3, k=3, n=3, seed=17, floaty=True, assigned=5),
        dict(rows=11, experts=4, k=2, n=3, seed=19, floaty=True),
    ]
    for combo in combos:
        _verify(solve, _case(**combo))
    x, weight, offs, grad_output = _case(rows=8, experts=4, k=3, n=2, seed=23, assigned=8)
    # int32 的 offs 与 int64 等价
    _verify(solve, (x, weight, offs.to(torch.int32), grad_output))
    # 非连续视图：靠转置构造，形状不变但 stride 变了
    wide = torch.cat([x, x], dim=1)[:, :x.size(1)]
    _verify(solve, (wide, weight, offs, grad_output))
    tall = torch.cat([weight, weight], dim=2)[:, :, :weight.size(2)]
    _verify(solve, (x, tall, offs, grad_output))
    _verify(solve, (x, weight, offs, grad_output.flip(0).contiguous().flip(0)))


def check_group_algebra_invariants(solve):
    # 组内行置换必须让 out / grad_input 同样置换，grad_weight 与 counts 完全不动
    x, weight, offs, grad_output = _case(rows=10, experts=4, k=3, n=2, seed=29, counts=[3, 0, 4, 3])
    out, grad_input, grad_weight, counts = _verify(solve, (x, weight, offs, grad_output))
    ends = offs.tolist()
    perm, start = [], 0
    for end in ends:
        perm.extend(range(end - 1, start - 1, -1))
        start = end
    index = torch.tensor(perm, dtype=torch.int64)
    shuffled = (x[index].contiguous(), weight, offs, grad_output[index].contiguous())
    out2, grad_input2, grad_weight2, counts2 = _verify(solve, shuffled)
    torch.testing.assert_close(out2, out[index], rtol=0, atol=0)
    torch.testing.assert_close(grad_input2, grad_input[index], rtol=0, atol=0)
    torch.testing.assert_close(grad_weight2, grad_weight, rtol=0, atol=0)
    assert torch.equal(counts2, counts)
    # 空专家的权重不可被读到：换成随机值，四个返回项逐位不变
    empty = [e for e in range(weight.size(0)) if counts[e].item() == 0]
    assert empty, "this case must contain at least one empty group"
    rng = torch.Generator().manual_seed(101)
    swapped = weight.clone()
    for e in empty:
        swapped[e] = torch.randn(weight.size(1), weight.size(2), generator=rng, dtype=torch.float64) * 7.0
    out3, grad_input3, grad_weight3, counts3 = _verify(solve, (x, swapped, offs, grad_output))
    torch.testing.assert_close(out3, out, rtol=0, atol=0)
    torch.testing.assert_close(grad_input3, grad_input, rtol=0, atol=0)
    torch.testing.assert_close(grad_weight3, grad_weight, rtol=0, atol=0)
    assert torch.equal(counts3, counts)
    assert bool((grad_weight[empty] == 0.0).all())
    # counts 是 offs 的差分，且和等于 offs 末值
    torch.testing.assert_close(counts.to(torch.float64), torch.cat(
        [offs.to(torch.float64)[:1], offs.to(torch.float64)[1:] - offs.to(torch.float64)[:-1]]),
        rtol=0, atol=0)
    assert int(counts.sum()) == int(offs[-1])


def check_all_empty_and_full_offsets(solve):
    rows, experts, k, n = 6, 4, 2, 3
    x, weight, _, grad_output = _case(rows=rows, experts=experts, k=k, n=n, seed=31, assigned=rows)
    # 先用一次「所有行都有主人」的分派把同形状的缓冲写满非零值再丢弃：靠 empty 起手
    # 的实现有很大概率从缓存分配器里拿到这段脏内存，从而把未初始化的尾部行暴露出来
    dirty = _case(rows=rows, experts=experts, k=k, n=n, seed=97, counts=[3, 3, 0, 0])
    solve(*dirty)
    del dirty
    # 全空：所有行都落在未分配的尾部
    zero_offs = torch.zeros(experts, dtype=torch.int64)
    out, grad_input, grad_weight, counts = _verify(solve, (x, weight, zero_offs, grad_output))
    assert bool((out == 0.0).all()) and bool((grad_input == 0.0).all())
    assert bool((grad_weight == 0.0).all()) and bool((counts == 0).all())
    # 只有最后一个专家吃下全部行，前面全空
    last_full = torch.tensor([0, 0, 0, rows], dtype=torch.int64)
    _verify(solve, (x, weight, last_full, grad_output))
    # 只有第一个专家吃下全部行
    first_full = torch.tensor([rows, rows, rows, rows], dtype=torch.int64)
    _verify(solve, (x, weight, first_full, grad_output))
    # 中间空、首尾非空
    sandwich = torch.tensor([2, 2, 2, rows], dtype=torch.int64)
    out, _, grad_weight, counts = _verify(solve, (x, weight, sandwich, grad_output))
    assert counts.tolist() == [2, 0, 0, rows - 2]
    assert bool((grad_weight[1:3] == 0.0).all())
    # 单专家
    _verify(solve, (x, weight[:1], torch.tensor([rows], dtype=torch.int64), grad_output))
    # int32 的 offs 同样合法
    _verify(solve, (x, weight, zero_offs.to(torch.int32), grad_output))


def check_tail_rows(solve):
    # 尾部行有非零输入和非零梯度，但它们既不能进 out/grad_input，也不能进任何 grad_weight
    x = torch.tensor([[1.0, 1.0], [2.0, 0.0], [0.0, 3.0], [5.0, 5.0], [-2.0, 4.0]], dtype=torch.float64)
    weight = torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[2.0, 1.0], [0.0, 1.0]]], dtype=torch.float64)
    grad_output = torch.tensor([[1.0, 0.0], [0.0, 1.0], [3.0, 3.0], [9.0, 9.0], [7.0, -1.0]],
                               dtype=torch.float64)
    # 先把同形状的缓冲填上非零值再丢弃，提高「用 empty 当输出」被暴露出来的概率
    dirty = _case(rows=5, experts=2, k=2, n=2, seed=77, counts=[3, 2])
    solve(*dirty)
    del dirty
    for offs in ([1, 2], [1, 1], [2, 2]):
        out, grad_input, grad_weight, counts = _verify(
            solve, (x, weight, torch.tensor(offs, dtype=torch.int64), grad_output))
        tail = out[offs[-1]:]
        assert bool((tail == 0.0).all()), offs
        assert bool((grad_input[offs[-1]:] == 0.0).all()), offs
        assert int(counts.sum()) == offs[-1]
    # 显式核对：e0 一行、e1 一行，后三行没主人
    out, grad_input, grad_weight, counts = _verify(
        solve, (x, weight, torch.tensor([1, 2], dtype=torch.int64), grad_output))
    torch.testing.assert_close(out, torch.tensor(
        [[1.0, 1.0], [4.0, 2.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=torch.float64), rtol=0, atol=0)
    torch.testing.assert_close(grad_input, torch.tensor(
        [[1.0, 0.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=torch.float64), rtol=0, atol=0)
    # grad_weight[0] = x0^T g0 = [[1,0],[1,0]]；grad_weight[1] = x1^T g1 = [[0,2],[0,0]]
    torch.testing.assert_close(grad_weight, torch.tensor(
        [[[1.0, 0.0], [1.0, 0.0]], [[0.0, 2.0], [0.0, 0.0]]], dtype=torch.float64), rtol=0, atol=0)
    assert counts.tolist() == [1, 1]


def check_exactness(solve):
    # 0.25 倍数的数据：BLAS 与纯 Python 标量乘加必须逐位相同，容差 0
    for seed in (41, 43, 47, 53):
        x, weight, offs, grad_output = _case(rows=9, experts=5, k=4, n=3, seed=seed, quarter=True)
        out, grad_input, grad_weight, counts = _verify(solve, (x, weight, offs, grad_output))
        expected = _oracle(x, weight, offs, grad_output)
        for actual, target in zip((out, grad_input, grad_weight), expected[:3]):
            torch.testing.assert_close(actual, target, rtol=0, atol=0)
        assert torch.equal(counts, expected[3])
    args = _example()
    out, grad_input, grad_weight, _ = _verify(solve, args)
    e_out, e_gi, e_gw, _ = _oracle(*args)
    torch.testing.assert_close(out, e_out, rtol=0, atol=0)
    torch.testing.assert_close(grad_input, e_gi, rtol=0, atol=0)
    torch.testing.assert_close(grad_weight, e_gw, rtol=0, atol=0)


def check_invalid(solve):
    x, weight, offs, grad_output = _case(rows=5, experts=3, k=2, n=2, seed=3, assigned=5)

    def expect_raises(args, note):
        try:
            solve(*args)
        except ValueError:
            return
        raise AssertionError(f"{note} should raise ValueError")

    expect_raises(("x", weight, offs, grad_output), "x is not a tensor")
    expect_raises((x.tolist(), weight, offs, grad_output), "x is a list")
    expect_raises((weight, weight, offs, grad_output), "x is 3-D")
    expect_raises((x.to(torch.float32), weight, offs, grad_output), "x float32")
    expect_raises((x, weight.to(torch.float32), offs, grad_output), "weight float32")
    expect_raises((x.reshape(-1), weight, offs, grad_output), "x 1-D")
    expect_raises((x.unsqueeze(0), weight, offs, grad_output), "x 3-D")
    expect_raises((torch.zeros(0, 2, dtype=torch.float64), weight,
                   torch.tensor([0, 0, 0], dtype=torch.int64), torch.zeros(0, 2, dtype=torch.float64)), "no rows")
    expect_raises((x, torch.zeros(0, 2, 2, dtype=torch.float64),
                   torch.zeros(0, dtype=torch.int64), grad_output), "no experts")
    expect_raises((x, weight.reshape(-1), offs, grad_output), "weight not 3-D")
    expect_raises((x, weight.unsqueeze(0), offs, grad_output), "weight 4-D")
    expect_raises((x, weight, offs.reshape(1, -1), grad_output), "offs 2-D")
    expect_raises((x, weight, torch.zeros(0, dtype=torch.int64), grad_output), "offs empty")
    expect_raises((x, weight, offs.to(torch.float64), grad_output), "offs float")
    expect_raises((x, weight, offs.to(torch.int16), grad_output), "offs int16")
    expect_raises((x, weight, torch.tensor([1, 2], dtype=torch.int64), grad_output), "offs length mismatch")
    expect_raises((x, weight, torch.tensor([-1, 2, 3], dtype=torch.int64), grad_output), "offs negative")
    expect_raises((x, weight, torch.tensor([3, 2, 5], dtype=torch.int64), grad_output), "offs decreasing")
    expect_raises((x, weight, torch.tensor([1, 2, 9], dtype=torch.int64), grad_output), "offs past last row")
    expect_raises((x, weight[:, :1].contiguous(), offs, grad_output), "k mismatch x/weight")
    expect_raises((x, weight, offs, grad_output[:, :1].contiguous()), "n mismatch weight/grad")
    expect_raises((x, weight, offs, grad_output[:3].contiguous()), "row count mismatch")
    expect_raises((x, weight, offs, grad_output.to(torch.float32)), "grad float32")
    expect_raises((x * float("nan"), weight, offs, grad_output), "nan x")
    expect_raises((x, weight * float("inf"), offs, grad_output), "inf weight")
    expect_raises((x, weight, offs, grad_output * float("nan")), "nan grad")
    # 合法边界：offs 全等、末值恰为行数、int32、单专家、大跨度量级
    solve(x, weight, torch.tensor([5, 5, 5], dtype=torch.int64), grad_output)
    solve(x, weight, offs.to(torch.int32), grad_output)
    solve(x, weight[:1], torch.tensor([5], dtype=torch.int64), grad_output)
    solve(x * 1e12, weight * 1e-6, offs, grad_output)


def check_maximum_shape(solve):
    _verify(solve, _case(rows=32, experts=8, k=4, n=4, seed=61, assigned=32))
    _verify(solve, _case(rows=32, experts=8, k=4, n=4, seed=67, quarter=True))
    _verify(solve, _case(rows=32, experts=16, k=2, n=2, seed=71))
