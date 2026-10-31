import math

import torch


def _softmax_row(row, tau):
    """纯标量 softmax：平移到最大值后逐项 exp，用 fsum 归一"""
    shifted = [v / tau for v in row]
    m = max(shifted)
    exps = [math.exp(v - m) for v in shifted]
    total = math.fsum(exps)
    return [v / total for v in exps]


def _div(num, den):
    """标量除法，如实复刻 IEEE：0/0 -> NaN，正数/0 -> ±inf，NaN 照常扩散"""
    if den == 0.0:
        if num == 0.0:
            return float('nan')
        return math.inf if num > 0.0 else -math.inf
    return num / den


def _add_eps(total, eps):
    return total + eps if eps else total


def _col_sum(q, j):
    return math.fsum(row[j] for row in q)


def _oracle(logits, iters, tau=1.0, square=False, eps=0.0):
    """纯 Python 列表实现：先列后行交替，eps 只加在除数上"""
    q = [_softmax_row(row, tau) for row in logits.tolist()]
    for _ in range(iters):
        cols = [_add_eps(_col_sum(q, j), eps) for j in range(len(q[0]))]
        q = [[_div(v, cols[j]) for j, v in enumerate(row)] for row in q]
        q = [[_div(v, _add_eps(math.fsum(row), eps)) for v in row] for row in q]
    if square:
        q = [[v * v for v in row] for row in q]
        q = [[_div(v, _add_eps(math.fsum(row), eps)) for v in row] for row in q]
    col_load = [_col_sum(q, j) for j in range(len(q[0]))]
    return (torch.tensor(q, dtype=torch.float64),
            torch.tensor(col_load, dtype=torch.float64))


def _case(t=4, e=3, seed=7, scale=3.0):
    rng = torch.Generator().manual_seed(seed)
    return torch.rand((t, e), generator=rng, dtype=torch.float64) * (2 * scale) - scale


def _verify(solve, args):
    tensors = [v for v in args if isinstance(v, torch.Tensor)]
    before = {id(v): v.clone() for v in tensors}
    expected = _oracle(*args)
    result = solve(*args)
    assert type(result) is tuple and len(result) == 2
    for actual, target in zip(result, expected):
        assert isinstance(actual, torch.Tensor) and actual.device.type == "cpu"
        assert actual.dtype == torch.float64 and actual.shape == target.shape, (
            actual.shape, target.shape)
        torch.testing.assert_close(actual, target, rtol=2e-11, atol=2e-11, equal_nan=True)
    for key, snapshot in before.items():
        for v in tensors:
            if id(v) == key:
                assert torch.equal(v, snapshot)
    return result


def _example():
    return torch.tensor([[0.0, math.log(2.0)], [0.0, math.log(2.0)]], dtype=torch.float64)


def check_example(solve):
    logits = _example()
    q, col = _verify(solve, (logits, 0))
    # softmax([0, ln 2]) = [1/3, 2/3]，两行相同 -> 列和 [2/3, 4/3]
    torch.testing.assert_close(q, torch.tensor([[1 / 3, 2 / 3], [1 / 3, 2 / 3]],
                                               dtype=torch.float64), rtol=0, atol=0)
    torch.testing.assert_close(col, torch.tensor([2 / 3, 4 / 3], dtype=torch.float64),
                               rtol=0, atol=0)
    # 一轮列归一化把两列都拉到 1，行归一化后每个元素都是 0.5，已经是定点
    q, col = _verify(solve, (logits, 1))
    torch.testing.assert_close(q, torch.full((2, 2), 0.5, dtype=torch.float64), rtol=0, atol=0)
    torch.testing.assert_close(col, torch.ones(2, dtype=torch.float64), rtol=0, atol=0)
    _verify(solve, (logits, 5))
    # square 把 [1/3, 2/3] 变成 [1/9, 4/9]，再按行归一化 -> [0.2, 0.8]
    q, col = _verify(solve, (logits, 0, 1.0, True, 0.0))
    torch.testing.assert_close(q[0], torch.tensor([0.2, 0.8], dtype=torch.float64),
                               rtol=0, atol=1e-15)
    torch.testing.assert_close(col, torch.tensor([0.4, 1.6], dtype=torch.float64),
                               rtol=0, atol=1e-15)
    # square 作用在已经是 0.5 的定点上没有效果
    q, _ = _verify(solve, (logits, 1, 1.0, True, 0.0))
    torch.testing.assert_close(q, torch.full((2, 2), 0.5, dtype=torch.float64), rtol=0, atol=0)


check_example.example = True


def check_matches_oracle(solve):
    for seed in range(1, 5):
        for iters in (0, 1, 3, 6):
            for tau in (0.25, 1.0, 4.0):
                for square in (False, True):
                    for eps in (0.0, 1e-20, 0.1):
                        _verify(solve, (_case(seed=seed, scale=[1.0, 3.0, 12.0][seed % 3]),
                                        iters, tau, square, eps))


def check_doubly_stochastic_convergence(solve):
    logits = torch.tensor([[0.0, 1.0, -1.0], [2.0, 0.0, -2.0], [0.0, 0.0, 1.0]],
                          dtype=torch.float64)
    # 行和：eps=0 时行归一化是精确的，列和才会一轮轮逼近 1
    prev_err = None
    for iters in range(1, 7):
        q, col = _verify(solve, (logits, iters))
        assert bool(torch.isfinite(q).all())
        torch.testing.assert_close(q.sum(dim=1), torch.ones(3, dtype=torch.float64),
                                   rtol=0, atol=1e-12)
        err = float((col - 1.0).abs().max())
        assert err < 0.2, (iters, col.tolist())
        if prev_err is not None:
            assert err < prev_err, (iters, prev_err, err)
        prev_err = err
    assert prev_err < 0.005, prev_err
    # eps 加在除数上会把行和系统性地压到 1 以下，且与行归一化前的行和相关
    q, col = _verify(solve, (logits, 3, 1.0, False, 0.1))
    rows = q.sum(dim=1)
    assert bool((rows > 0).all()) and bool((rows < 1.0).all()), rows.tolist()
    assert float((col - 1.0).abs().max()) > 0.05, col.tolist()
    # 列置换等变：交换两个专家，Q 与列负载同步换列。
    # 容差不是 0：行归一化的分母是沿专家轴归约，换列会改变浮点求和的结合顺序。
    perm = torch.tensor([2, 0, 1], dtype=torch.int64)
    base_q, base_col = solve(logits, 4)
    perm_q, perm_col = solve(logits[:, perm], 4)
    torch.testing.assert_close(perm_q, base_q[:, perm], rtol=0, atol=1e-14)
    torch.testing.assert_close(perm_col, base_col[perm], rtol=0, atol=1e-14)
    # token 置换等变，且列和与行顺序无关
    row_perm = torch.tensor([2, 0, 1], dtype=torch.int64)
    rp_q, rp_col = solve(logits[row_perm], 4)
    torch.testing.assert_close(rp_q, base_q[row_perm], rtol=0, atol=1e-14)
    torch.testing.assert_close(rp_col, base_col, rtol=1e-12, atol=1e-12)
    # 每行加自己的常数不改变任何结果
    shift = torch.tensor([[3.0], [-5.0], [1.5]], dtype=torch.float64)
    sh_q, sh_col = solve(logits + shift, 4)
    torch.testing.assert_close(sh_q, base_q, rtol=0, atol=1e-15)
    torch.testing.assert_close(sh_col, base_col, rtol=0, atol=1e-15)


def check_column_collapse(solve):
    # 两个 token 都把概率压在专家 1、2 上，专家 0 整列下溢成 0
    logits = torch.tensor([[-800.0, 0.0, 0.0], [-900.0, 0.0, 0.0]], dtype=torch.float64)
    q, col = _verify(solve, (logits, 0))
    torch.testing.assert_close(q, torch.tensor([[0.0, 0.5, 0.5], [0.0, 0.5, 0.5]],
                                               dtype=torch.float64), rtol=0, atol=0)
    torch.testing.assert_close(col, torch.tensor([0.0, 1.0, 1.0], dtype=torch.float64),
                               rtol=0, atol=0)
    # eps=0：归一化一个全零列得到 0/0，NaN 会顺着行和扩散到整行
    q, col = _verify(solve, (logits, 1))
    assert bool(torch.isnan(q).all()), q.tolist()
    assert bool(torch.isnan(col).all()), col.tolist()
    # eps>0：0 除以 eps 仍是 0，死列被如实报告为 0 负载而不是被抬起来
    q, col = _verify(solve, (logits, 1, 1.0, False, 1e-30))
    torch.testing.assert_close(q, torch.tensor([[0.0, 0.5, 0.5], [0.0, 0.5, 0.5]],
                                               dtype=torch.float64), rtol=0, atol=1e-15)
    torch.testing.assert_close(col, torch.tensor([0.0, 1.0, 1.0], dtype=torch.float64),
                               rtol=0, atol=1e-15)
    # 死列 + 平方支路：[0, 1/3, 2/3] 平方归一化后是 [0, 1/5, 4/5]
    one = torch.tensor([[-800.0, 0.0, math.log(2.0)]], dtype=torch.float64)
    q, col = _verify(solve, (one, 0, 1.0, True, 1e-20))
    torch.testing.assert_close(q[0], torch.tensor([0.0, 0.2, 0.8], dtype=torch.float64),
                               rtol=0, atol=1e-15)
    torch.testing.assert_close(col, torch.tensor([0.0, 0.2, 0.8], dtype=torch.float64),
                               rtol=0, atol=1e-15)


def check_degenerate_shapes(solve):
    # 单 token：列归一化把每个元素都变成 1，行归一化再把它们摊成均匀分布。
    # 注意列和也是 1/E 而不是 1：一轮以行归一化收尾，T=1 时总质量只有 1，
    # 摊到 E=4 个专家上就不可能让每列都到 1——这正是「行约束精确、列约束只是逼近」。
    logits = torch.tensor([[0.0, 1.0, -1.0, 2.0]], dtype=torch.float64)
    q, col = _verify(solve, (logits, 2))
    torch.testing.assert_close(q, torch.full((1, 4), 0.25, dtype=torch.float64),
                               rtol=0, atol=1e-15)
    torch.testing.assert_close(col, torch.full((4,), 0.25, dtype=torch.float64),
                               rtol=0, atol=1e-15)
    q, col = _verify(solve, (logits, 1))
    torch.testing.assert_close(q, torch.full((1, 4), 0.25, dtype=torch.float64),
                               rtol=0, atol=1e-15)
    # 单专家：无论多少 token，行和恒为 1，列和等于 token 数
    logits = torch.tensor([[0.0], [5.0], [-5.0], [100.0]], dtype=torch.float64)
    for iters in (0, 1, 4, 16):
        q, col = _verify(solve, (logits, iters))
        torch.testing.assert_close(q, torch.ones(4, 1, dtype=torch.float64), rtol=0, atol=0)
        torch.testing.assert_close(col, torch.tensor([4.0], dtype=torch.float64), rtol=0, atol=0)
    # 全零 logits 本来就是双随机矩阵的定点
    z = torch.zeros(3, 3, dtype=torch.float64)
    q, col = _verify(solve, (z, 6))
    torch.testing.assert_close(q, torch.full((3, 3), 1 / 3, dtype=torch.float64), rtol=0, atol=0)
    torch.testing.assert_close(col, torch.ones(3, dtype=torch.float64), rtol=0, atol=0)
    # 大轮数不会溢出或出现负数
    big = _case(t=6, e=5, seed=99, scale=20.0)
    q, col = _verify(solve, (big, 16, 0.25, True, 1e-20))
    assert bool((q >= 0).all()) and bool(torch.isfinite(q).all())
    assert abs(float(q.sum()) - float(q.shape[0])) <= 1e-9
    # 非连续输入也要能算
    nc = _case(t=4, e=3, seed=13)
    _verify(solve, (nc.t().contiguous().t(), 3))


def check_invalid(solve):
    logits = _case()
    for bad_iters in (-1, 17, 1.0, True, False, None, [1], "1"):
        try:
            solve(logits, bad_iters)
        except ValueError:
            pass
        else:
            raise AssertionError(f"iters={bad_iters!r} should raise ValueError")
    for bad_tau in (0.0, -1.0, float('nan'), float('inf'), True, "1.0", None):
        try:
            solve(logits, 1, bad_tau)
        except ValueError:
            pass
        else:
            raise AssertionError(f"tau={bad_tau!r} should raise ValueError")
    for bad_eps in (-1e-3, float('nan'), float('inf'), True, "0.1", None):
        try:
            solve(logits, 1, 1.0, False, bad_eps)
        except ValueError:
            pass
        else:
            raise AssertionError(f"eps={bad_eps!r} should raise ValueError")
    for bad_square in (0, 1, None, "True", []):
        try:
            solve(logits, 1, 1.0, bad_square)
        except ValueError:
            pass
        else:
            raise AssertionError(f"square={bad_square!r} should raise ValueError")
    for bad_logits in ([0.0, 1.0], 3, None, torch.rand(4, dtype=torch.float64),
                       torch.rand(2, 2, 2, dtype=torch.float64), torch.rand(2, 3),
                       torch.zeros(0, 3, dtype=torch.float64),
                       torch.zeros(3, 0, dtype=torch.float64),
                       torch.randint(0, 2, (2, 3), dtype=torch.int64)):
        try:
            solve(bad_logits, 1)
        except ValueError:
            pass
        else:
            raise AssertionError("logits should raise ValueError")
    # 合法的标量取值：int 与 bool 之外的组合，iters=0，eps=0
    solve(logits, 0, 1, False, 0)


def check_maximum_shape(solve):
    for iters, square, eps in ((16, True, 0.0), (16, False, 1e-20), (0, True, 0.5)):
        _verify(solve, (_case(t=8, e=8, seed=41, scale=8.0), iters, 0.5, square, eps))
