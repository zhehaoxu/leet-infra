import torch
import torch.nn.functional as F


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic(solve):
    indices = torch.tensor([1, 3, 2])
    y = solve(indices)
    expected = [[0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
    assert y.shape == (3, 4), f"类别数应为 max+1=4，got {tuple(y.shape)}"
    assert torch.equal(y, torch.tensor(expected)), f"got {y.tolist()}"
    assert y.dtype == torch.long, "one-hot 应为 int64"


def check_matches_torch(solve):
    torch.manual_seed(0)
    indices = torch.randint(0, 10, (5, 4))
    assert torch.equal(solve(indices), F.one_hot(indices))
    assert torch.equal(solve(indices, 16), F.one_hot(indices, 16)), "显式 num_classes 应与 torch 一致"


def check_extra_classes_zero(solve):
    # num_classes 大于 max+1 时，多出的列全 0
    indices = torch.tensor([0, 1])
    y = solve(indices, 5)
    assert y.shape == (2, 5)
    assert torch.equal(y[:, 2:], torch.zeros(2, 3)), "超出类别应补零"


def check_2d_input(solve):
    indices = torch.tensor([[1, 0], [2, 2]])
    y = solve(indices, 3)
    assert y.shape == (2, 2, 3)
    assert (y.sum(-1) == 1).all(), "每行恰好一个 1"
    assert torch.equal(y, F.one_hot(indices, 3))


def check_all_onehot_rows_sum_to_one(solve):
    indices = torch.randint(0, 7, (100,))
    y = solve(indices, 7)
    assert (y.sum(-1) == 1).all() and (y.argmax(-1) == indices).all()


check_basic.example = True
