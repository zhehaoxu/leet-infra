import torch


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic(solve):
    a = torch.tensor([0.0, 0.0, 2.0, 2.0])
    b = torch.tensor([1.0, 1.0, 3.0, 3.0])
    y = solve(a, b)
    assert abs(y.item() - 1.0 / 7.0) < 1e-5, y.item()


def check_identical_is_one(solve):
    box = torch.tensor([[1.0, 2, 5, 9]])
    assert torch.allclose(solve(box, box), torch.ones(1))


def check_disjoint_zero(solve):
    a = torch.tensor([0.0, 0.0, 1.0, 1.0])
    b = torch.tensor([5.0, 5.0, 6.0, 6.0])
    assert solve(a, b).item() == 0.0


def check_batched(solve):
    a = torch.tensor([[0.0, 0, 2, 2], [0, 0, 1, 1]])
    b = torch.tensor([[1.0, 1, 3, 3], [2, 2, 3, 3]])
    y = solve(a, b)
    assert abs(y[0].item() - 1.0 / 7.0) < 1e-5
    assert y[1].item() == 0.0


def check_containment(solve):
    # 小框完全在大框内：IoU = 小/大
    big = torch.tensor([0.0, 0.0, 4.0, 4.0])   # 16
    small = torch.tensor([1.0, 1.0, 2.0, 2.0])  # 1, 交集=1
    assert abs(solve(big, small).item() - 1.0 / 16.0) < 1e-5


check_basic.example = True
