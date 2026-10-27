import itertools

import torch

# 示例期望值（实测回填）：input=[10,20,30,40], src=[1,2,3,4,5], index=[0,1,1,3,3]
EX_INPUT = [10.0, 20.0, 30.0, 40.0]
EX_SRC = [1.0, 2.0, 3.0, 4.0, 5.0]
EX_INDEX = [0, 1, 1, 3, 3]
EXAMPLE_SUM_INC = [11.0, 25.0, 30.0, 49.0]
EXAMPLE_SUM_NOINC = [1.0, 5.0, 30.0, 9.0]
EXAMPLE_PROD_INC = [10.0, 120.0, 30.0, 800.0]
EXAMPLE_PROD_NOINC = [1.0, 6.0, 30.0, 20.0]
EXAMPLE_MEAN_INC = [5.5, 8.333333, 30.0, 16.333334]
EXAMPLE_MEAN_NOINC = [1.0, 2.5, 30.0, 4.5]
EXAMPLE_AMAX_NOINC = [1.0, 3.0, 30.0, 5.0]
EXAMPLE_AMIN_INC = [1.0, 2.0, 30.0, 4.0]

_REDUCES = ("sum", "prod", "mean", "amax", "amin")


def check_example(solve):
    inp = torch.tensor(EX_INPUT)
    src = torch.tensor(EX_SRC)
    idx = torch.tensor(EX_INDEX)
    expect = {
        ("sum", True): EXAMPLE_SUM_INC,
        ("sum", False): EXAMPLE_SUM_NOINC,
        ("prod", True): EXAMPLE_PROD_INC,
        ("prod", False): EXAMPLE_PROD_NOINC,
        ("mean", True): EXAMPLE_MEAN_INC,
        ("mean", False): EXAMPLE_MEAN_NOINC,
        ("amax", False): EXAMPLE_AMAX_NOINC,
        ("amin", True): EXAMPLE_AMIN_INC,
    }
    for (red, inc), exp in expect.items():
        got = solve(inp, src, idx, reduce=red, include_self=inc)
        assert got.shape == inp.shape and got.dtype == inp.dtype, (red, inc, got)
        assert torch.allclose(got, torch.tensor(exp), atol=1e-6), (red, inc, got.tolist())
    # solve 不得修改 input
    snapshot = inp.clone()
    solve(inp, src, idx, reduce="mean")
    assert torch.equal(inp, snapshot), "solve 修改了 input"


check_example.example = True


def check_mean_welford_semantics(solve):
    """mean 的增量式（Welford 型）更新口径：批量「总和/总数」公式会给出不同结果。"""
    zero = torch.zeros(1)
    # base=0, 贡献 [1,2,3,4]：Welford 步进 = 2.0，batch 公式 = 2.5
    got = solve(zero, torch.tensor([1.0, 2.0, 3.0, 4.0]),
                torch.zeros(4, dtype=torch.long), reduce="mean", include_self=True)
    assert torch.allclose(got, torch.tensor([2.0]), atol=1e-6), got.tolist()
    # base=0, 贡献 [1e8,1,1,1]：f32 Welford 步进 = 2e7，batch 公式 = 2.5e7
    got = solve(zero, torch.tensor([1e8, 1.0, 1.0, 1.0]),
                torch.zeros(4, dtype=torch.long), reduce="mean", include_self=True)
    assert torch.allclose(got, torch.tensor([20000000.0]), atol=1e-5, rtol=1e-5), got.tolist()
    # base=1e8, 贡献 [1,1,1,1]：inc=True -> 2e7；inc=False -> 1.0（batch 公式给 2.5e7）
    big = torch.tensor([1e8])
    ones = torch.tensor([1.0, 1.0, 1.0, 1.0])
    idx4 = torch.zeros(4, dtype=torch.long)
    got = solve(big, ones, idx4, reduce="mean", include_self=True)
    assert torch.allclose(got, torch.tensor([20000000.0]), atol=1e-5, rtol=1e-5), got.tolist()
    got = solve(big, ones, idx4, reduce="mean", include_self=False)
    assert torch.allclose(got, torch.tensor([1.0]), atol=1e-6), got.tolist()
    # 与原生对拍（全部上述情形）
    for base, sv, inc in ((0.0, [1.0, 2.0, 3.0, 4.0], True),
                          (0.0, [1e8, 1.0, 1.0, 1.0], True),
                          (1e8, [1.0, 1.0, 1.0, 1.0], True),
                          (1e8, [1.0, 1.0, 1.0, 1.0], False)):
        inp = torch.tensor([base])
        exp = inp.clone().scatter_reduce_(0, torch.zeros(len(sv), dtype=torch.long),
                                          torch.tensor(sv), reduce="mean", include_self=inc)
        got = solve(inp, torch.tensor(sv), torch.zeros(len(sv), dtype=torch.long),
                    reduce="mean", include_self=inc)
        assert torch.allclose(got, exp, atol=1e-5, rtol=1e-5), (base, inc, got.item(), exp.item())


def check_matches_native_all_combos(solve):
    """主 oracle：5 种 reduce x include_self 两态 x dim{0,1,-1} 全组合对拍原生。"""
    for seed in range(6):
        for reduce in _REDUCES:
            for inc in (True, False):
                # dim=0：index 与 input 同形
                torch.manual_seed(100 + seed)
                inp = torch.randn(3, 4) * 3.0
                idx = torch.randint(0, 3, (3, 4))
                src = torch.randn(3, 4) * 2.0
                exp = inp.clone().scatter_reduce_(0, idx, src, reduce=reduce, include_self=inc)
                got = solve(inp, src, idx, dim=0, reduce=reduce, include_self=inc)
                assert torch.allclose(got, exp, atol=1e-5, rtol=1e-5), (reduce, inc, 0, seed)
                # dim=1：index 在 dim 维比 input 宽（合法），含重复坐标
                torch.manual_seed(200 + seed)
                inp = torch.randn(3, 4) * 3.0
                idx = torch.randint(0, 4, (3, 5))
                src = torch.randn(3, 5) * 2.0
                exp = inp.clone().scatter_reduce_(1, idx, src, reduce=reduce, include_self=inc)
                got = solve(inp, src, idx, dim=1, reduce=reduce, include_self=inc)
                assert torch.allclose(got, exp, atol=1e-5, rtol=1e-5), (reduce, inc, 1, seed)
                # dim=-1：负维索引
                torch.manual_seed(300 + seed)
                inp = torch.randn(4, 3) * 3.0
                idx = torch.randint(0, 3, (4, 6))
                src = torch.randn(4, 6) * 2.0
                exp = inp.clone().scatter_reduce_(-1, idx, src, reduce=reduce, include_self=inc)
                got = solve(inp, src, idx, dim=-1, reduce=reduce, include_self=inc)
                assert torch.allclose(got, exp, atol=1e-5, rtol=1e-5), (reduce, inc, -1, seed)
    # 3-D：index 在 dim 维比 input 窄
    for seed in range(2):
        for reduce in _REDUCES:
            for inc in (True, False):
                torch.manual_seed(400 + seed)
                inp = torch.randn(2, 3, 4)
                idx = torch.randint(0, 3, (2, 2, 4))
                src = torch.randn(2, 2, 4)
                exp = inp.clone().scatter_reduce_(1, idx, src, reduce=reduce, include_self=inc)
                got = solve(inp, src, idx, dim=1, reduce=reduce, include_self=inc)
                assert torch.allclose(got, exp, atol=1e-5, rtol=1e-5), (reduce, inc, seed)
    # src 比 index 大：同坐标配对，多余元素不参与
    for seed in range(2):
        torch.manual_seed(500 + seed)
        inp = torch.randn(3, 4)
        idx = torch.randint(0, 4, (3, 4))
        src = torch.randn(3, 7) * 2.0
        for reduce in ("sum", "mean"):
            for inc in (True, False):
                exp = inp.clone().scatter_reduce_(1, idx, src, reduce=reduce, include_self=inc)
                got = solve(inp, src, idx, dim=1, reduce=reduce, include_self=inc)
                assert torch.allclose(got, exp, atol=1e-5, rtol=1e-5), (reduce, inc, seed)


def _nav(t, coords):
    for c in coords:
        t = t[c]
    return t


def _oracle_loop(input, src, index, dim, reduce, include_self):
    """独立 python 循环 oracle：product 逐坐标 + dict 状态累加，与解法的
    排序分组路径完全不同。"""
    ndim = input.dim()
    d = dim % ndim
    inp_l, src_l, idx_l = input.tolist(), src.tolist(), index.tolist()
    state = {}
    for coords in itertools.product(*(range(s) for s in index.shape)):
        i = _nav(idx_l, coords)
        v = _nav(src_l, coords)
        pos = coords[:d] + (i,) + coords[d + 1:]
        if pos not in state:
            if include_self:
                state[pos] = [_nav(inp_l, pos), 1]  # 初值 = input 原值，随后仍要并入 v
            else:
                state[pos] = [v, 1]  # 首个贡献即初值
                continue
        cur, n = state[pos]
        n += 1
        if reduce == "sum":
            cur = cur + v
        elif reduce == "prod":
            cur = cur * v
        elif reduce == "amax":
            cur = max(cur, v)
        elif reduce == "amin":
            cur = min(cur, v)
        else:  # mean：Welford 型
            cur = cur + (v - cur) / n
        state[pos] = [cur, n]
    vals = []
    for pos in itertools.product(*(range(s) for s in input.shape)):
        vals.append(state[pos][0] if pos in state else _nav(inp_l, pos))
    return torch.tensor(vals, dtype=input.dtype).reshape(input.shape)


def check_python_oracle_sum_amax_mean(solve):
    """独立 python 循环 oracle 对拍 sum/amax/mean 三个组合（两种 include_self）。"""
    for seed in range(3):
        for reduce in ("sum", "amax", "mean"):
            for inc in (True, False):
                for dim in (0, 1):
                    torch.manual_seed(700 + seed)
                    inp = torch.randn(3, 4) * 3.0
                    idx = torch.randint(0, 3 if dim == 0 else 4, (3, 4))
                    src = torch.randn(3, 4) * 2.0
                    got = solve(inp, src, idx, dim=dim, reduce=reduce, include_self=inc)
                    exp = _oracle_loop(inp, src, idx, dim, reduce, inc)
                    assert torch.allclose(got, exp, atol=1e-5, rtol=1e-5), \
                        (reduce, inc, dim, seed, got.tolist(), exp.tolist())


def check_empty_index(solve):
    inp = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]])
    for inc in (True, False):
        got = solve(inp, torch.zeros(0, 4), torch.zeros(0, 4, dtype=torch.long),
                    dim=0, include_self=inc)
        assert torch.equal(got, inp), "空 index 时结果应恒等于 input"
        got = solve(inp, torch.zeros(3, 0), torch.zeros(3, 0, dtype=torch.long),
                    dim=1, include_self=inc)
        assert torch.equal(got, inp)
    got = solve(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]), torch.zeros(0),
                torch.zeros(0, dtype=torch.long), include_self=False)
    assert torch.equal(got, torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]))


def check_full_coverage(solve):
    """index 恰好覆盖每个位置一次：每个位置只有单个贡献，各 reduce 结果应一致，
    且与原生 scatter_reduce 完全对齐。"""
    inp = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    idx = torch.tensor([[0, 1, 0], [1, 0, 1]])  # 每列都是 0/1 的排列 -> 6 个位置各被触及一次
    src = torch.tensor([[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]])
    ref = None
    for reduce in _REDUCES:
        for inc in (True, False):
            got = solve(inp, src, idx, dim=0, reduce=reduce, include_self=inc)
            exp = inp.clone().scatter_reduce_(0, idx, src, reduce=reduce, include_self=inc)
            assert torch.equal(got, exp), (reduce, inc, got.tolist(), exp.tolist())
            if not inc:
                # 单贡献位置上 sum/prod/mean/amax/amin 都应等于该贡献本身
                if ref is None:
                    ref = got
                assert torch.equal(got, ref), (reduce, got.tolist(), ref.tolist())


def check_zero_input(solve):
    inp = torch.zeros(3)
    idx = torch.tensor([0, 0, 1])
    src = torch.tensor([2.0, 3.0, 4.0])
    exp = {("prod", True): [0.0, 0.0, 0.0], ("prod", False): [6.0, 4.0, 0.0],
           ("sum", False): [5.0, 4.0, 0.0], ("amax", False): [3.0, 4.0, 0.0],
           ("mean", False): [2.5, 4.0, 0.0]}
    for (red, inc), ev in exp.items():
        got = solve(inp, src, idx, reduce=red, include_self=inc)
        assert torch.allclose(got, torch.tensor(ev), atol=1e-6), (red, inc, got.tolist())
        native = inp.clone().scatter_reduce_(0, idx, src, reduce=red, include_self=inc)
        assert torch.allclose(got, native, atol=1e-6), (red, inc)


def check_prod_first_contribution(solve):
    """prod 的 include_self=False 首贡献语义：从 src 起算，不乘 input 原值。"""
    inp = torch.tensor([5.0, 7.0])
    idx = torch.tensor([0, 0, 1])
    src = torch.tensor([2.0, 3.0, 4.0])
    got = solve(inp, src, idx, reduce="prod", include_self=False)
    assert torch.equal(got, torch.tensor([6.0, 4.0])), got.tolist()
    got = solve(inp, src, idx, reduce="prod", include_self=True)
    assert torch.equal(got, torch.tensor([30.0, 28.0])), got.tolist()
    # sum 对照
    got = solve(inp, src, idx, reduce="sum", include_self=False)
    assert torch.equal(got, torch.tensor([5.0, 4.0])), got.tolist()
    got = solve(inp, src, idx, reduce="sum", include_self=True)
    assert torch.equal(got, torch.tensor([10.0, 11.0])), got.tolist()


def check_mean_count_semantics(solve):
    """mean include_self=False 只平均被 scatter 的值；未触及位置保持 input 原值。"""
    inp = torch.tensor([6.0, -3.0])
    idx = torch.tensor([0, 0, 0])
    src = torch.tensor([2.0, 4.0, 9.0])
    got = solve(inp, src, idx, reduce="mean", include_self=False)
    assert torch.allclose(got, torch.tensor([5.0, -3.0]), atol=1e-6), got.tolist()
    got = solve(inp, src, idx, reduce="mean", include_self=True)
    # Welford: 6 <- 4 <- 4 <- 5.25；位置 1 未触及保持 -3
    assert torch.allclose(got, torch.tensor([5.25, -3.0]), atol=1e-6), got.tolist()
    native = inp.clone().scatter_reduce_(0, idx, src, reduce="mean", include_self=False)
    assert torch.allclose(solve(inp, src, idx, reduce="mean", include_self=False),
                          native, atol=1e-6)


def check_float64(solve):
    torch.manual_seed(900)
    inp = torch.randn(3, 4, dtype=torch.float64) * 3.0
    idx = torch.randint(0, 3, (3, 5))
    src = torch.randn(3, 5, dtype=torch.float64) * 2.0
    for reduce in ("sum", "mean", "amax"):
        for inc in (True, False):
            exp = inp.clone().scatter_reduce_(1, idx, src, reduce=reduce, include_self=inc)
            got = solve(inp, src, idx, dim=1, reduce=reduce, include_self=inc)
            assert got.dtype == torch.float64
            assert torch.allclose(got, exp, atol=1e-9, rtol=1e-9), (reduce, inc)


def check_invalid_reduce_and_oob(solve):
    inp = torch.zeros(3, 4)
    src = torch.ones(3, 4)
    idx = torch.zeros(3, 4, dtype=torch.long)
    try:
        solve(inp, src, idx, reduce="median")
    except ValueError:
        pass
    else:
        raise AssertionError("非法 reduce 应抛 ValueError")
    try:
        solve(inp, src, torch.full((3, 4), -1, dtype=torch.long))
    except RuntimeError:
        pass
    else:
        raise AssertionError("负 index 应抛 RuntimeError（与原生一致）")
    try:
        solve(inp, src, torch.full((3, 4), 99, dtype=torch.long))
    except RuntimeError:
        pass
    else:
        raise AssertionError("越界 index 应抛 RuntimeError（与原生一致）")
    try:
        solve(inp, src, idx, dim=5)
    except IndexError:
        pass
    else:
        raise AssertionError("dim 越界应抛 IndexError（与原生一致）")
