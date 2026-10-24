import torch


def _ref(inputs):
    """独立重算：Python 逐元素累加，不复用 stack().sum()。"""
    acc = torch.zeros_like(inputs[0])
    for t in inputs:
        acc = acc + t
    return acc


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    inputs = [torch.tensor([[1., 2.], [3., 4.]]), torch.tensor([[10., 20.], [30., 40.]])]
    out = solve(inputs)
    want = torch.tensor([[11., 22.], [33., 44.]])
    assert len(out) == 2, "应返回每 rank 一份完整结果"
    for r in out:
        assert torch.equal(r, want), r.tolist()


def check_all_ranks_identical_and_full(solve):
    torch.manual_seed(42)
    inputs = [torch.randn(3, 4) for _ in range(5)]
    out = solve(inputs)
    total = _ref(inputs)
    assert len(out) == 5
    for r in out:
        assert r.shape == (3, 4) and torch.allclose(r, total), "每 rank 都应是完整逐元素和"


def check_equals_gather_of_scatter(solve):
    # All-Reduce == 先 Reduce-Scatter（各 rank 一段和）再 All-Gather（拼回完整和）
    torch.manual_seed(7)
    L, w = 6, 3
    inputs = [torch.randn(L) for _ in range(w)]
    total = torch.stack(inputs, 0).sum(0)
    chunk = L // w
    scattered = [total[i * chunk:(i + 1) * chunk] for i in range(w)]   # reduce-scatter 结果
    gathered = torch.cat(scattered, dim=0)                             # all-gather 拼回
    out = solve(inputs)
    for r in out:
        assert torch.equal(r, gathered) and torch.equal(gathered, total)


def check_integer_exact(solve):
    inputs = [torch.tensor([1, 2, 3]), torch.tensor([10, 20, 30]), torch.tensor([100, 200, 300])]
    out = solve(inputs)
    for r in out:
        assert torch.equal(r, torch.tensor([111, 222, 333])), r.tolist()


def check_single_rank(solve):
    x = torch.randn(4)
    out = solve([x])
    assert len(out) == 1 and torch.equal(out[0], x)


def check_input_not_mutated(solve):
    inputs = [torch.randn(2, 3) for _ in range(4)]
    snapshot = [t.clone() for t in inputs]
    solve(inputs)
    for a, b in zip(inputs, snapshot):
        assert torch.equal(a, b), "不应修改输入张量"


check_example.example = True
