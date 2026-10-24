import torch


# 独立参考：手动逐元素拼接出完整结果
def _ref(inputs, dst):
    flat = []
    for t in inputs:
        flat.extend(t.reshape(-1).tolist())
    full = torch.tensor(flat, dtype=inputs[0].dtype)
    return [full if j == dst else inputs[j].clone() for j in range(len(inputs))]


# 样例用例
def check_example(solve):
    inputs = [torch.tensor([1., 2.]), torch.tensor([3., 4.]), torch.tensor([5., 6.])]
    out = solve(inputs, 1)
    assert len(out) == 3
    assert torch.equal(out[0], torch.tensor([1., 2.])), out[0].tolist()
    assert torch.equal(out[1], torch.tensor([1., 2., 3., 4., 5., 6.])), out[1].tolist()
    assert torch.equal(out[2], torch.tensor([5., 6.])), out[2].tolist()


def check_only_dst_has_full(solve):
    # 只有 dst 拿全体，其余必须等于自己原片（区别于 all-gather）
    torch.manual_seed(42)
    inputs = [torch.randn(3) for _ in range(4)]
    dst = 2
    out = solve(inputs, dst)
    full = torch.cat(inputs, dim=0)
    assert torch.equal(out[dst], full)
    for j in range(4):
        if j != dst:
            assert torch.equal(out[j], inputs[j]), j


def check_various_dst(solve):
    torch.manual_seed(7)
    inputs = [torch.randn(2) for _ in range(3)]
    for dst in range(3):
        out = solve(inputs, dst)
        ref = _ref(inputs, dst)
        for j in range(3):
            assert torch.equal(out[j], ref[j]), (dst, j)


def check_dst_zero_default(solve):
    inputs = [torch.tensor([7., 8.]), torch.tensor([9., 10.])]
    out = solve(inputs, 0)
    assert torch.equal(out[0], torch.tensor([7., 8., 9., 10.]))
    assert torch.equal(out[1], torch.tensor([9., 10.])), "非目标 rank 保留原片"


def check_single_rank(solve):
    x = torch.tensor([1., 2., 3.])
    out = solve([x], 0)
    assert len(out) == 1 and torch.equal(out[0], x)


def check_input_not_mutated(solve):
    inputs = [torch.tensor([1., 2.]), torch.tensor([3., 4.]), torch.tensor([5., 6.])]
    snap = [t.clone() for t in inputs]
    solve(inputs, 1)
    for a, b in zip(inputs, snap):
        assert torch.equal(a, b), "不应修改输入张量"


check_example.example = True
