import torch


def _oracle(order, num_replicas, rank, drop_last):
    """纯 Python：effective 用取模下标直接构造，不做任何列表拼接，也不借用切片步长"""
    size = order.shape[0]
    indices = order.tolist()
    if drop_last and size % num_replicas != 0:
        num_samples = size // num_replicas
    else:
        num_samples = (size + num_replicas - 1) // num_replicas
    total_size = num_samples * num_replicas

    effective = []
    for slot in range(total_size):
        if drop_last:
            effective.append(indices[slot])
        else:
            effective.append(indices[slot % size])

    shard = []
    step = rank
    while step < total_size:
        shard.append(effective[step])
        step += num_replicas

    coverage = [0] * size
    for value in effective:
        coverage[value] += 1

    return (torch.tensor(shard, dtype=torch.int64), num_samples, total_size,
            torch.tensor(coverage, dtype=torch.int64))


def _case(size=7, replicas=3, rank=0, drop_last=False, seed=1):
    rng = torch.Generator().manual_seed(seed)
    order = torch.randperm(size, generator=rng)
    return order, replicas, rank, drop_last


def _example():
    order = torch.tensor([6, 2, 0, 4, 1, 3, 5], dtype=torch.int64)
    return order, 3, 0, False


def _verify(solve, args):
    order, replicas, rank, drop_last = args
    snapshot = order.clone()
    expected = _oracle(order, replicas, rank, drop_last)
    result = solve(order, replicas, rank, drop_last)
    assert isinstance(result, tuple) and len(result) == 4, f"expected a 4-tuple, got {type(result)}"
    shard, num_samples, total_size, coverage = result
    assert isinstance(num_samples, int) and not isinstance(num_samples, bool), "num_samples must be a python int"
    assert isinstance(total_size, int) and not isinstance(total_size, bool), "total_size must be a python int"
    assert torch.equal(shard, expected[0]), ("shard", shard.tolist(), expected[0].tolist())
    assert num_samples == expected[1], (num_samples, expected[1])
    assert total_size == expected[2], (total_size, expected[2])
    assert coverage.dtype == torch.int64 and coverage.shape == expected[3].shape, (
        coverage.dtype, coverage.shape)
    assert torch.equal(coverage, expected[3]), ("coverage", coverage.tolist(), expected[3].tolist())
    assert shard.dtype == torch.int64
    assert torch.equal(order, snapshot)
    return result


def check_example(solve):
    order = _example()[0]
    # drop_last=False：ceil(7/3)=3、total=9、补齐头两个 (6, 2)
    shard, num_samples, total_size, coverage = _verify(solve, (order, 3, 0, False))
    assert shard.tolist() == [6, 4, 5]
    assert (num_samples, total_size) == (3, 9)
    assert coverage.tolist() == [1, 1, 2, 1, 1, 1, 2]
    assert int(coverage.sum()) == total_size
    # 另外两个 rank 的分片
    assert _verify(solve, (order, 3, 1, False))[0].tolist() == [2, 1, 6]
    assert _verify(solve, (order, 3, 2, False))[0].tolist() == [0, 3, 2]
    # drop_last=True：floor(7/3)=2、total=6、样本 5 被丢掉
    shard, num_samples, total_size, coverage = _verify(solve, (order, 3, 0, True))
    assert shard.tolist() == [6, 4]
    assert (num_samples, total_size) == (2, 6)
    assert coverage.tolist() == [1, 1, 1, 1, 1, 0, 1]
    assert _verify(solve, (order, 3, 1, True))[0].tolist() == [2, 1]
    assert _verify(solve, (order, 3, 2, True))[0].tolist() == [0, 3]


check_example.example = True


def check_matches_oracle(solve):
    combos = [
        (1, 1, 0, False), (1, 4, 0, False), (1, 4, 3, False), (1, 4, 0, True), (1, 4, 2, True),
        (2, 5, 1, False), (3, 2, 0, False), (3, 2, 1, False), (4, 8, 5, False), (5, 3, 2, True),
        (6, 6, 0, False), (6, 6, 5, False), (7, 3, 0, False), (7, 3, 1, True), (8, 3, 2, False),
        (9, 4, 1, False), (10, 7, 6, True), (11, 5, 3, False), (12, 5, 4, True), (13, 4, 0, False),
        (16, 9, 8, False), (17, 6, 5, True), (23, 8, 7, False), (24, 12, 11, False), (31, 9, 4, True),
    ]
    for size, replicas, rank, drop_last in combos:
        for seed in (1, 2, 3):
            _verify(solve, _case(size=size, replicas=replicas, rank=rank,
                                 drop_last=drop_last, seed=seed))
    # 排列本身也可以由 randperm 之外的方式给出
    order = torch.tensor([3, 1, 0, 2], dtype=torch.int64)
    for rank in range(3):
        _verify(solve, (order, 3, rank, False))


def check_shards_interleave_to_effective(solve):
    for size, replicas, drop_last, seed in [(10, 4, False, 5), (10, 4, True, 5), (7, 3, False, 6),
                                            (13, 5, False, 7), (4, 8, False, 8), (9, 2, True, 9)]:
        order, _, _, _ = _case(size=size, replicas=replicas, rank=0, drop_last=drop_last, seed=seed)
        _, num_samples, total_size, coverage = _verify(solve, (order, replicas, 0, drop_last))
        assert num_samples * replicas == total_size
        shards = [_verify(solve, (order, replicas, rank, drop_last))[0].tolist()
                  for rank in range(replicas)]
        assert all(len(shard) == num_samples for shard in shards), shards
        # 交织回去必须还原 effective；effective 又由 order 与取模规则唯一决定
        merged = [shards[rank][step] for step in range(num_samples) for rank in range(replicas)]
        assert len(merged) == total_size
        if drop_last and size % replicas != 0:
            assert merged == order.tolist()[:total_size]
        else:
            assert merged == [order.tolist()[slot % size] for slot in range(total_size)]
        # coverage 是全局账：每个下标的出现次数 == 它在 merged 里的次数
        expected = [merged.count(value) for value in range(size)]
        assert coverage.tolist() == expected, (coverage.tolist(), expected)
        assert int(coverage.sum()) == total_size


def check_padding_and_truncation_edges(solve):
    order = torch.tensor([4, 2, 9, 1, 7, 0, 3, 8, 6, 5], dtype=torch.int64)
    # padding == 0：整除，补齐与砍尾给出同一个序列
    for rank in range(5):
        padded = _verify(solve, (order, 5, rank, False))
        truncated = _verify(solve, (order, 5, rank, True))
        assert torch.equal(padded[0], truncated[0]), rank
        assert padded[1] == truncated[1] and padded[2] == truncated[2], rank
        assert torch.equal(padded[3], truncated[3]), rank
    # padding == N：total_size = 2N，恰好把整个 order 复制一遍
    order5 = torch.tensor([4, 2, 0, 1, 3], dtype=torch.int64)
    shard, num_samples, total_size, coverage = _verify(solve, (order5, 10, 0, False))
    assert (num_samples, total_size) == (1, 10)
    assert coverage.tolist() == [2, 2, 2, 2, 2]
    assert shard.tolist() == [4]
    # padding > N：N < R，需要把整个 order 抄好几遍
    order3 = torch.tensor([1, 0, 2], dtype=torch.int64)
    for rank in range(5):
        single, num_samples, total_size, coverage = _verify(solve, (order3, 5, rank, False))
        assert (num_samples, total_size) == (1, 5) and len(single) == 1
    # effective = [1, 0, 2, 1, 0]：样本 0、1 各两次，样本 2 一次
    assert coverage.tolist() == [2, 2, 1]
    assert _verify(solve, (order3, 5, 0, False))[0].tolist() == [1]
    assert _verify(solve, (order3, 5, 4, False))[0].tolist() == [0]
    # N < R 且砍尾：一步都不跑，但返回的仍是空张量
    empty, num_samples, total_size, coverage = _verify(solve, (order3, 5, 0, True))
    assert empty.numel() == 0 and empty.dtype == torch.int64
    assert (num_samples, total_size) == (0, 0)
    assert bool((coverage == 0).all())
    # N == 1：唯一的样本被复制 R 遍
    only = torch.tensor([0], dtype=torch.int64)
    shard, num_samples, total_size, coverage = _verify(solve, (only, 4, 2, False))
    assert shard.tolist() == [0] and num_samples == 1 and total_size == 4 and coverage.tolist() == [4]
    # R == 1：拿到整条 effective
    shard, num_samples, total_size, coverage = _verify(solve, (order, 1, 0, False))
    assert shard.tolist() == order.tolist() and total_size == 10 and bool((coverage == 1).all())


def check_duplicated_samples_are_the_head(solve):
    # 不砍尾时，被读两次的样本恰是 order 的头 padding 个，而不是「最后一个 rank 多拿一个」
    for size, replicas, seed in [(11, 4, 21), (13, 5, 22), (17, 6, 23), (19, 7, 24)]:
        order, _, _, _ = _case(size=size, replicas=replicas, rank=0, drop_last=False, seed=seed)
        _, num_samples, total_size, coverage = _verify(solve, (order, replicas, 0, False))
        padding = total_size - size
        assert padding == replicas * num_samples - size
        head = order.tolist()[:padding]
        counts = {}
        for value in range(size):
            counts[value] = int(coverage[value])
        assert all(counts[value] == (2 if value in head else 1) for value in range(size)), (
            size, replicas, coverage.tolist(), head)
        # 砍尾时少的恰好是 effective 之外的尾巴
        _, _, total_kept, coverage_cut = _verify(solve, (order, replicas, 0, True))
        dropped = [int(value) for value in order.tolist()[total_kept:]]
        kept = [int(value) for value in order.tolist()[:total_kept]]
        assert coverage_cut[dropped].tolist() == [0] * len(dropped)
        assert coverage_cut[kept].tolist() == [1] * len(kept)
        assert int(coverage_cut.sum()) == total_kept


def check_identity_order_is_strided(solve):
    # order = 0..N-1 时最能看出「跨步切片」和「连续块」的区别
    size, replicas = 12, 4
    order = torch.arange(size, dtype=torch.int64)
    for rank in range(replicas):
        shard, num_samples, total_size, coverage = _verify(solve, (order, replicas, rank, False))
        assert shard.tolist() == list(range(rank, size, replicas)), (rank, shard.tolist())
        assert num_samples == 3 and total_size == 12 and bool((coverage == 1).all())
    # 除不尽：连续块答法会把多出来的槽给最后一个 rank，跨步切片把它们分给了头两个 rank
    order = torch.arange(10, dtype=torch.int64)
    shards = [_verify(solve, (order, 4, rank, False))[0].tolist() for rank in range(4)]
    assert shards == [[0, 4, 8], [1, 5, 9], [2, 6, 0], [3, 7, 1]], shards


def check_invalid(solve):
    order = _case(size=8, replicas=3, rank=0, seed=4)[0]

    def expect_raises(args, note):
        try:
            solve(*args)
        except ValueError:
            return
        raise AssertionError(f"{note} should raise ValueError")

    expect_raises(([0, 1, 2], 2, 0, False), "order is a list")
    expect_raises((torch.tensor([[0, 1], [2, 3]], dtype=torch.int64), 2, 0, False), "order 2-D")
    expect_raises((torch.zeros(0, dtype=torch.int64), 2, 0, False), "order empty")
    expect_raises((order.to(torch.int32), 3, 0, False), "order int32")
    expect_raises((order.to(torch.float64), 3, 0, False), "order float")
    expect_raises((order[:4], 3, 0, False), "not a permutation (duplicates/missing)")
    expect_raises((torch.tensor([0, 1, 2, 3, 4, 5, 6, 8], dtype=torch.int64), 3, 0, False), "index past N")
    expect_raises((torch.tensor([-1, 0, 1, 2, 3, 4, 5, 6], dtype=torch.int64), 3, 0, False), "negative index")
    expect_raises((order, 0, 0, False), "num_replicas zero")
    expect_raises((order, -1, 0, False), "num_replicas negative")
    expect_raises((order, 3.0, 0, False), "num_replicas float")
    expect_raises((order, True, 0, False), "num_replicas bool")
    expect_raises((order, 3, 3, False), "rank == num_replicas")
    expect_raises((order, 3, -1, False), "rank negative")
    expect_raises((order, 3, 1.0, False), "rank float")
    expect_raises((order, 3, 0, 0), "drop_last is not a bool")
    expect_raises((order, 3, 0, None), "drop_last None")
    # 合法边界：R == 1、rank 取到上限、N == 1、drop_last 两种
    solve(order, 1, 0, True)
    solve(order, 8, 7, False)
    solve(torch.tensor([0], dtype=torch.int64), 1, 0, True)
    solve(order, 5, 4, True)


def check_maximum_shape(solve):
    _verify(solve, _case(size=512, replicas=7, rank=6, drop_last=False, seed=31))
    _verify(solve, _case(size=512, replicas=7, rank=0, drop_last=True, seed=31))
    _verify(solve, _case(size=257, replicas=64, rank=63, drop_last=False, seed=33))
    _verify(solve, _case(size=257, replicas=64, rank=1, drop_last=True, seed=33))
