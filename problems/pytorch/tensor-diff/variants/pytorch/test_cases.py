import torch

# 示例：DX = [[1,4,9,16],[2,3,5,9]]（实跑参考答案回填）
EXAMPLE_X = [[1.0, 4.0, 9.0, 16.0], [2.0, 3.0, 5.0, 9.0]]
EXAMPLE_N1 = [[3.0, 5.0, 7.0], [1.0, 2.0, 4.0]]
EXAMPLE_N2 = [[2.0, 2.0], [1.0, 2.0]]
EXAMPLE_DIM0 = [[1.0, -1.0, -4.0, -7.0]]
EXAMPLE_PREPEND = [[1.0, 3.0, 5.0, 7.0], [2.0, 1.0, 2.0, 4.0]]
EXAMPLE_APPEND = [[3.0, 5.0, 7.0, -16.0], [1.0, 2.0, 4.0, -9.0]]


def check_example(solve):
    x = torch.tensor(EXAMPLE_X)
    assert torch.allclose(solve(x), torch.tensor(EXAMPLE_N1), atol=1e-6)
    assert torch.allclose(solve(x, n=2), torch.tensor(EXAMPLE_N2), atol=1e-6)
    assert torch.allclose(solve(x, dim=0), torch.tensor(EXAMPLE_DIM0), atol=1e-6)
    assert torch.allclose(solve(x, prepend=torch.zeros(2, 1)),
                          torch.tensor(EXAMPLE_PREPEND), atol=1e-6)
    assert torch.allclose(solve(x, append=torch.zeros(2, 1)),
                          torch.tensor(EXAMPLE_APPEND), atol=1e-6)


check_example.example = True


def check_matches_torch(solve):
    torch.manual_seed(701)
    x = torch.randn(3, 7)
    # 只取差分后非空的组合（长度为 0 的退化情形由 check_invalid_args 按题面口径断言）
    for n, dimv in [(0, -1), (1, -1), (2, -1), (3, -1), (6, -1),
                    (0, 0), (1, 0), (2, 0), (1, 1), (4, 1)]:
        ref = torch.diff(x, n=n, dim=dimv)
        got = solve(x, n=n, dim=dimv)
        assert got.shape == ref.shape, (n, dimv, got.shape, ref.shape)
        assert torch.allclose(got, ref, atol=1e-6), (n, dimv)
        assert got.dtype == x.dtype
    # prepend / append 各种组合
    p = torch.randn(3, 2)
    a = torch.randn(3, 1)
    for pv, av in [(p, None), (None, a), (p, a)]:
        ref = torch.diff(x, prepend=pv, append=av)
        got = solve(x, prepend=pv, append=av)
        assert got.shape == ref.shape
        assert torch.allclose(got, ref, atol=1e-6)


def check_length_and_semantics(solve):
    # 输出长度 = 拼接后长度 − n，且二阶差分等于一阶差分的差分
    torch.manual_seed(702)
    x = torch.randn(4, 9)
    assert solve(x, dim=1).shape == (4, 8)
    assert solve(x, n=3, dim=1).shape == (4, 6)
    assert solve(x, prepend=torch.zeros(4, 5), dim=1).shape == (4, 13)
    assert solve(x, append=torch.zeros(4, 2), dim=1).shape == (4, 10)
    once = solve(x, dim=1)
    assert torch.allclose(solve(x, n=2, dim=1), solve(once, n=1, dim=1), atol=1e-6)
    assert torch.allclose(solve(x, n=0), x, atol=1e-6)
    # 线性序列的一阶差分是常数、二阶差分全 0
    lin = torch.arange(0.0, 10.0).reshape(2, 5)
    assert torch.allclose(solve(lin), torch.full((2, 4), 1.0), atol=1e-6)
    assert torch.allclose(solve(lin, n=2), torch.zeros(2, 3), atol=1e-6)


def check_middle_axis_dims(solve):
    torch.manual_seed(703)
    x = torch.randn(2, 3, 4)
    for dimv in [0, 1, 2, -1, -2, -3]:
        ref = torch.diff(x, dim=dimv)
        got = solve(x, dim=dimv)
        assert got.shape == ref.shape, dimv
        assert torch.allclose(got, ref, atol=1e-6), dimv
    # 中间轴差分：相邻元素沿该轴切换
    g = solve(x, dim=1)
    assert torch.allclose(g[:, 0, :], x[:, 1, :] - x[:, 0, :], atol=1e-6)
    assert torch.allclose(g[:, -1, :], x[:, 2, :] - x[:, 1, :], atol=1e-6)


def check_invalid_args(solve):
    x = torch.randn(3, 7)
    for kwargs in [dict(n=-1), dict(n=1.5), dict(dim=7), dict(dim=-8)]:
        try:
            solve(x, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"{kwargs} 应抛 ValueError")
    # prepend/append 形状不符：除 dim 轴外必须与 x 完全一致
    for kwargs in [dict(prepend=torch.randn(2, 3)), dict(append=torch.randn(4)),
                   dict(prepend=torch.randn(3, 7, 1))]:
        try:
            solve(x, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"{kwargs} 应抛 ValueError")
    # 差分后长度非正要报错，而不是交回一个空张量
    try:
        solve(torch.randn(3, 2), n=3, dim=1)
    except ValueError:
        pass
    else:
        raise AssertionError("长度非正应抛 ValueError")
