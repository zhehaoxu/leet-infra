import torch


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic(solve):
    x = torch.tensor([[1.0, 2, 3], [4, 5, 6]])
    assert solve(x, 1).tolist() == [[1.0, 3.0, 6.0], [4.0, 9.0, 15.0]]
    assert solve(x, 0).tolist() == [[1.0, 2.0, 3.0], [5.0, 7.0, 9.0]]


def check_negative_axis(solve):
    torch.manual_seed(0)
    x = torch.randn(2, 3, 4)
    assert torch.allclose(solve(x, -1), x.cumsum(-1))


def check_matches_cumsum(solve):
    torch.manual_seed(1)
    x = torch.randn(3, 5, 2)
    for ax in range(3):
        assert torch.allclose(solve(x, ax), x.cumsum(ax), atol=1e-5)


def check_last_is_total(solve):
    # 累积和的最后一个元素应等于整轴求和
    torch.manual_seed(2)
    x = torch.randn(4, 6)
    y = solve(x, 1)
    assert torch.allclose(y[:, -1], x.sum(1), atol=1e-5)


check_basic.example = True
