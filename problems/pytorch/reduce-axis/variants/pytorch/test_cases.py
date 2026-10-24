import torch


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic(solve):
    x = torch.tensor([[1.0, 2, 3], [4, 5, 6]])
    assert solve(x, 1).tolist() == [6.0, 15.0]
    assert solve(x, 0).tolist() == [5.0, 7.0, 9.0]


def check_keepdim(solve):
    x = torch.tensor([[1.0, 2, 3], [4, 5, 6]])
    y = solve(x, 1, True)
    assert y.shape == (2, 1)
    assert y.tolist() == [[6.0], [15.0]]


def check_negative_axis(solve):
    torch.manual_seed(0)
    x = torch.randn(2, 3, 4)
    assert torch.allclose(solve(x, -1), x.sum(-1))
    assert torch.allclose(solve(x, -2), x.sum(-2))


def check_matches_sum(solve):
    torch.manual_seed(1)
    x = torch.randn(3, 5, 2, 4)
    for ax in range(4):
        assert torch.allclose(solve(x, ax), x.sum(ax), atol=1e-5)
        assert torch.allclose(solve(x, ax, True), x.sum(ax, keepdim=True), atol=1e-5)


check_basic.example = True
