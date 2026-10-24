import torch


def _ref(inputs, root):
    """独立重算：Python 顺序累加求全局和，只写到 root，其余原样。"""
    acc = torch.zeros_like(inputs[0])
    for t in inputs:
        acc = acc + t
    out = [t.clone() for t in inputs]
    out[root] = acc
    return out


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    inputs = [torch.tensor([[1., 2.], [3., 4.]]), torch.tensor([[10., 20.], [30., 40.]])]
    out = solve(inputs, root=0)
    assert len(out) == 2
    assert torch.equal(out[0], torch.tensor([[11., 22.], [33., 44.]])), out[0].tolist()
    assert torch.equal(out[1], torch.tensor([[10., 20.], [30., 40.]])), "非根 rank 应保持原值"


def check_only_root_has_sum(solve):
    torch.manual_seed(42)
    W = 5
    inputs = [torch.randn(3, 4) for _ in range(W)]
    root = 2
    out = solve(inputs, root)
    total = _ref(inputs, root)[root]
    assert torch.allclose(out[root], total), "root 应是完整逐元素和"
    for r in range(W):
        if r != root:
            assert torch.allclose(out[r], inputs[r]), "非根 rank 不应被改写"


def check_root_last(solve):
    inputs = [torch.tensor([1, 2]), torch.tensor([3, 4]), torch.tensor([5, 6])]
    out = solve(inputs, root=2)
    assert torch.equal(out[2], torch.tensor([9, 12])), out[2].tolist()
    assert torch.equal(out[0], torch.tensor([1, 2]))
    assert torch.equal(out[1], torch.tensor([3, 4]))


def check_single_rank(solve):
    x = torch.randn(4)
    out = solve([x], root=0)
    assert len(out) == 1 and torch.allclose(out[0], x)


def check_input_not_mutated(solve):
    inputs = [torch.randn(2, 3) for _ in range(4)]
    snapshot = [t.clone() for t in inputs]
    solve(inputs, root=1)
    for a, b in zip(inputs, snapshot):
        assert torch.equal(a, b), "不应修改输入张量"


check_example.example = True
