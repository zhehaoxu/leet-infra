import torch


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic(solve):
    a = torch.tensor([[1.0, 2.0]])
    b = torch.tensor([[3.0, 4.0], [5.0, 6.0]])
    y = solve([a, b], dim=0)
    assert y.shape == (3, 2), f"沿 dim0 拼接行数相加，got {tuple(y.shape)}"
    assert torch.equal(y, torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])), f"got {y.tolist()}"


def check_dim1(solve):
    torch.manual_seed(0)
    a, b, c = torch.randn(2, 3), torch.randn(2, 4), torch.randn(2, 1)
    assert torch.equal(solve([a, b, c], 1), torch.cat([a, b, c], dim=1))


def check_rank_preserved(solve):
    # cat 不新增维（与 stack 的关键区别）
    a = torch.randn(2, 3)
    y = solve([a, a], dim=0)
    assert y.dim() == 2, "cat 应保持维数不变，不像 stack 增维"
    assert y.shape == (4, 3)


def check_negative_dim(solve):
    torch.manual_seed(1)
    a = torch.randn(2, 3, 4)
    b = torch.randn(2, 3, 5)
    assert torch.equal(solve([a, b], -1), torch.cat([a, b], -1)), "负 dim 应等价最后一维"


def check_order_matters(solve):
    a = torch.tensor([[1.0]])
    b = torch.tensor([[2.0]])
    assert torch.equal(solve([a, b], 0), torch.tensor([[1.0], [2.0]]))
    assert not torch.equal(solve([a, b], 0), solve([b, a], 0)), "拼接顺序应影响结果"


def check_3d(solve):
    torch.manual_seed(2)
    parts = [torch.randn(2, 3, 1) for _ in range(4)]
    assert torch.equal(solve(parts, 2), torch.cat(parts, dim=2))


check_basic.example = True
