import torch


# 独立参考：逐 rank 列表拼接（与 solution 的 reshape-transpose 走不同路径）
def _ref(inputs):
    W = len(inputs)
    flat = [t.reshape(-1) for t in inputs]
    k = flat[0].numel() // W
    out = []
    for j in range(W):
        out.append(torch.cat([flat[q][j * k:(j + 1) * k] for q in range(W)], dim=0))
    return out


# 样例用例
def check_example(solve):
    inputs = [torch.tensor([10., 20, 30, 40]), torch.tensor([50., 60, 70, 80])]
    out = solve(inputs)
    assert len(out) == 2
    assert torch.equal(out[0], torch.tensor([10., 20, 50, 60])), out[0].tolist()
    assert torch.equal(out[1], torch.tensor([30., 40, 70, 80])), out[1].tolist()


def check_matches_ref_random(solve):
    for W, k in [(3, 2), (4, 3), (2, 5), (5, 1)]:
        torch.manual_seed(W * 13 + k)
        inputs = [torch.randn(W * k) for _ in range(W)]
        out = solve(inputs)
        ref = _ref(inputs)
        assert len(out) == W
        for j in range(W):
            assert out[j].shape == (W * k,)
            assert torch.equal(out[j], ref[j]), (W, k, j, out[j].tolist(), ref[j].tolist())


def check_element_conservation(solve):
    # 全元素多重集守恒：输出的所有元素 == 输入的所有元素（只是重排）
    torch.manual_seed(7)
    inputs = [torch.arange(9.).float() + q * 100 for q in range(3)]
    out = solve(inputs)
    got = torch.cat(out).sort().values
    exp = torch.cat([t.reshape(-1) for t in inputs]).sort().values
    assert torch.equal(got, exp)


def check_degenerates_to_allgather(solve):
    # 每 rank 的 W 个分片都相同 -> 每个 rank 收到「各 rank 的那一份」拼接 == All-Gather
    shards = [torch.tensor([1., 2.]), torch.tensor([3., 4.]), torch.tensor([5., 6.])]
    W = 3
    inputs = [torch.cat([s for _ in range(W)]) for s in shards]  # rank q: [sq,sq,sq]
    out = solve(inputs)
    # out[j] = cat_q inputs[q][j 片] = cat_q shards[q]，与 j 无关 -> 三行都等于 all-gather
    gathered = torch.cat(shards, dim=0)
    for j in range(W):
        assert torch.equal(out[j], gathered), (j, out[j].tolist(), gathered.tolist())


def check_single_rank(solve):
    x = torch.tensor([1., 2., 3.])
    out = solve([x])
    assert len(out) == 1 and torch.equal(out[0], x)


def check_input_not_mutated(solve):
    inputs = [torch.tensor([1., 2, 3, 4]), torch.tensor([5., 6, 7, 8])]
    snap = [t.clone() for t in inputs]
    solve(inputs)
    for a, b in zip(inputs, snap):
        assert torch.equal(a, b), "不应修改输入张量"


check_example.example = True
