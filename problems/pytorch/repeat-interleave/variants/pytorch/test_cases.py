import torch

# 示例：RX = [[1,2],[3,4]]（实跑参考答案回填）
EXAMPLE_X = [[1.0, 2.0], [3.0, 4.0]]
EXAMPLE_FLAT = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]
EXAMPLE_DIM0 = [[1.0, 2.0], [1.0, 2.0], [3.0, 4.0], [3.0, 4.0]]
EXAMPLE_VARIED = [[1.0, 2.0], [3.0, 4.0], [3.0, 4.0], [3.0, 4.0]]
EXAMPLE_ZERO = [5.0, 5.0, 7.0]


def check_example(solve):
    x = torch.tensor(EXAMPLE_X)
    assert solve(x, 2).shape == (8,)
    assert torch.equal(solve(x, 2), torch.tensor(EXAMPLE_FLAT))
    assert torch.equal(solve(x, 2, dim=0), torch.tensor(EXAMPLE_DIM0))
    assert torch.equal(solve(x, torch.tensor([1, 3]), dim=0), torch.tensor(EXAMPLE_VARIED))
    # repeats 里出现 0：该位置整个消失
    assert torch.equal(solve(torch.tensor([5.0, 6.0, 7.0]), torch.tensor([2, 0, 1])),
                       torch.tensor(EXAMPLE_ZERO))


check_example.example = True


def check_interleave_not_tile(solve):
    # 关键区分：interleave 是 [1,1,2,2,3,3]，tile/repeat 是 [1,2,3,1,2,3]
    x = torch.tensor([1.0, 2.0, 3.0])
    out = solve(x, 2)
    assert torch.equal(out, torch.tensor([1.0, 1.0, 2.0, 2.0, 3.0, 3.0]))
    assert not torch.equal(out, x.repeat(2))
    # 逐位置次数不同
    assert torch.equal(solve(x, torch.tensor([1, 2, 3])),
                       torch.tensor([1.0, 2.0, 2.0, 3.0, 3.0, 3.0]))
    assert solve(x, torch.tensor([1, 2, 3])).shape == (6,)


def check_matches_torch(solve):
    torch.manual_seed(801)
    x = torch.randn(3, 4)
    for dimv in [None, 0, 1, -1, -2]:
        count = x.numel() if dimv is None else x.shape[dimv]
        for rp in [1, 2, 5, torch.randint(0, 4, (count,), dtype=torch.long),
                   torch.ones(count, dtype=torch.int32),
                   torch.zeros(count, dtype=torch.long)]:
            ref = (torch.repeat_interleave(x, rp) if dimv is None
                   else torch.repeat_interleave(x, rp, dim=dimv))
            got = solve(x, rp, dim=dimv)
            assert got.shape == ref.shape, (dimv, rp if isinstance(rp, int) else "tensor")
            assert torch.equal(got, ref), (dimv, rp if isinstance(rp, int) else "tensor")
            assert got.dtype == x.dtype


def check_dim_none_flattens(solve):
    torch.manual_seed(802)
    x = torch.randn(2, 3, 4)
    # dim=None：先展平再逐元素重复，输出恒为 1-D
    out = solve(x, 3)
    assert out.dim() == 1 and out.shape[0] == x.numel() * 3
    assert torch.equal(out, x.reshape(-1).repeat_interleave(3))
    reps = torch.randint(1, 4, (x.numel(),), dtype=torch.long)
    assert torch.equal(solve(x, reps), x.reshape(-1).repeat_interleave(reps))
    # 沿中间轴：其余维保持原样，只有该轴变成 Σrepeats
    y = torch.randn(2, 4, 5)
    r = torch.tensor([2, 0, 1, 3], dtype=torch.long)
    got = solve(y, r, dim=1)
    assert got.shape == (2, 6, 5)
    assert torch.equal(got[:, 0, :], y[:, 0, :])
    assert torch.equal(got[:, 1, :], y[:, 0, :])
    assert torch.equal(got[:, 2, :], y[:, 2, :])       # index1 重复 0 次，直接跳到 index2
    assert torch.equal(got[:, 3, :], y[:, 3, :])
    assert torch.equal(got[:, -1, :], y[:, 3, :])


def check_output_is_copy(solve):
    # index_select 语义：输出是新内存，改输出不得影响输入
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    out = solve(x, 2, dim=0)
    assert out.data_ptr() != x.data_ptr()
    out.mul_(10.0)
    assert torch.equal(x, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    # 全 0 重复：长度为 0 的空张量，但形状轴数与 dtype 保留
    empty = solve(x, torch.zeros(2, dtype=torch.long), dim=0)
    assert empty.shape == (0, 2) and empty.dtype == x.dtype


def check_invalid_args(solve):
    x = torch.randn(3, 4)
    for rp in [torch.tensor([1, -2, 3, 4]), torch.tensor([1, 2]), torch.tensor([[1, 2], [3, 4]]),
               torch.rand(4), -1]:
        try:
            solve(x, rp, dim=0)
        except ValueError:
            continue
        raise AssertionError(f"repeats={rp} 应抛 ValueError")
    for dimv in [5, -4]:
        try:
            solve(x, 2, dim=dimv)
        except ValueError:
            continue
        raise AssertionError(f"dim={dimv} 应抛 ValueError")
