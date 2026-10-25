import torch


# 独立重算参考：结构不同——先把每个全局下标按「落在哪个 [b_r,b_{r+1}) 区间」逐位归属（区间成员扫描），
# 再统计每 rank 元素数；参数是否跨界用「其覆盖下标是否出现 >1 个 owner」判断，而非用边界不等式。
def _oracle(numels, world):
    numels = [int(x) for x in numels]
    total = sum(numels)
    b = [(total * r) // world for r in range(world + 1)]
    owner = []
    for x in range(total):
        r = world - 1
        while not (b[r] <= x < b[r + 1]):
            r -= 1
        owner.append(r)
    per_rank = [0] * world
    for o in owner:
        per_rank[o] += 1
    straddle = []
    s = 0
    for i, n in enumerate(numels):
        e = s + n
        if n > 0 and len(set(owner[s:e])) > 1:
            straddle.append(i)
        s = e
    return torch.tensor(per_rank, dtype=torch.long), torch.tensor(straddle, dtype=torch.long)


def check_example(solve):
    per_rank, straddle = solve([3, 4, 2, 5, 6], 4)
    assert torch.equal(per_rank, torch.tensor([5, 5, 5, 5], dtype=torch.long)), per_rank
    assert torch.equal(straddle, torch.tensor([1, 3, 4], dtype=torch.long)), straddle


def check_world_one_no_boundary(solve):
    for numels in ([3, 4, 2, 5, 6], [10, 20, 5], [7]):
        per_rank, straddle = solve(numels, 1)
        assert torch.equal(per_rank, torch.tensor([sum(numels)], dtype=torch.long)), per_rank
        assert straddle.numel() == 0, straddle


def check_aligned_no_straddle(solve):
    # 每个参数恰好铺满一个分片，边界只压在参数端点上（不算严格内部）→ 无跨界
    per_rank, straddle = solve([6, 6, 6, 6], 4)
    assert torch.equal(per_rank, torch.tensor([6, 6, 6, 6], dtype=torch.long)), per_rank
    assert straddle.numel() == 0, straddle


def check_single_param_straddle_many(solve):
    # 单个参数跨过多个边界：total=5,world=3 → b=[0,1,3,5], per_rank=[1,2,2]，参数 [0,5) 内含 1、3
    per_rank, straddle = solve([5], 3)
    assert torch.equal(per_rank, torch.tensor([1, 2, 2], dtype=torch.long)), per_rank
    assert torch.equal(straddle, torch.tensor([0], dtype=torch.long)), straddle


def check_matches_oracle(solve):
    cases = [
        ([3, 4, 2, 5, 6], 4),
        ([3, 4, 2, 5, 6], 5),
        ([1, 1, 1, 1, 1, 1, 1], 3),
        ([10, 20, 5, 15], 6),
        ([0, 5, 0, 5, 3], 4),
        ([100], 7),
        ([7, 7, 7], 2),
        ([2, 4, 6, 8, 10, 12], 5),
    ]
    for numels, w in cases:
        got = solve(numels, w)
        ref = _oracle(numels, w)
        assert torch.equal(got[0], ref[0]), (numels, w, got[0], ref[0])
        assert torch.equal(got[1], ref[1]), (numels, w, got[1], ref[1])


check_example.example = True
