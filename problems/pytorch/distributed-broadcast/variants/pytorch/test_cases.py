import torch


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    inputs = [torch.tensor([99.]), torch.tensor([99.]), torch.tensor([1., 2., 3.])]
    out = solve(inputs, 2)
    src = torch.tensor([1., 2., 3.])
    assert len(out) == 3
    for r in out:
        assert torch.equal(r, src), r.tolist()


def check_src_index_used(solve):
    # 源不是 0 时必须取对应 rank 的数据，其余占位不用
    inputs = [torch.tensor([0.]), torch.tensor([5., 5.]), torch.tensor([9.])]
    for s in range(3):
        out = solve(inputs, s)
        for r in out:
            assert torch.equal(r, inputs[s]), (s, r.tolist())


def check_2d_and_length(solve):
    torch.manual_seed(42)
    src_tensor = torch.randn(2, 3)
    inputs = [torch.zeros(1), src_tensor, torch.zeros(7), torch.zeros(2)]
    out = solve(inputs, 1)
    assert len(out) == 4
    for r in out:
        assert r.shape == (2, 3) and torch.equal(r, src_tensor)


def check_single_rank(solve):
    x = torch.tensor([3., 1., 4.])
    out = solve([x], 0)
    assert len(out) == 1 and torch.equal(out[0], x)


def check_input_not_mutated(solve):
    inputs = [torch.tensor([1.]), torch.tensor([2., 2.]), torch.tensor([3.])]
    snapshot = [t.clone() for t in inputs]
    solve(inputs, 1)
    for a, b in zip(inputs, snapshot):
        assert torch.equal(a, b), "不应修改输入张量"


check_example.example = True
