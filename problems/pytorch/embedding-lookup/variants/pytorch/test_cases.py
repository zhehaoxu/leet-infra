import torch


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic(solve):
    weight = torch.arange(12, dtype=torch.float32).reshape(4, 3)  # V=4, D=3
    indices = torch.tensor([0, 3, 1, 3])
    y = solve(weight, indices, padding_idx=3)
    expected = [[0.0, 1.0, 2.0], [0.0, 0.0, 0.0], [3.0, 4.0, 5.0], [0.0, 0.0, 0.0]]
    assert y.shape == (4, 3)
    assert torch.equal(y, torch.tensor(expected)), f"got {y.tolist()}"


def check_no_padding(solve):
    weight = torch.randn(5, 4)
    indices = torch.tensor([4, 0, 2, 2])
    y = solve(weight, indices)
    assert torch.equal(y, weight[[4, 0, 2, 2]]), "无 padding_idx 应等价按行取值"


def check_batched_indices(solve):
    weight = torch.randn(6, 3)
    indices = torch.tensor([[1, 2], [3, 3]])
    y = solve(weight, indices)
    assert y.shape == (2, 2, 3), f"形状应 (*indices, D)，got {tuple(y.shape)}"
    assert torch.equal(y[1, 1], weight[3])


def check_padding_zero_only(solve):
    # padding 行必须全 0，其余不受影响
    weight = torch.full((4, 3), 7.0)
    indices = torch.tensor([0, 2, 2])
    y = solve(weight, indices, padding_idx=2)
    assert torch.equal(y[0], torch.full((3,), 7.0)), "非 padding 行应保持"
    assert torch.equal(y[1], torch.zeros(3)) and torch.equal(y[2], torch.zeros(3)), "padding 行应置零"


def check_float64(solve):
    weight = torch.randn(3, 2, dtype=torch.float64)
    indices = torch.tensor([0, 1, 1, 2])
    y = solve(weight, indices)
    assert y.dtype == torch.float64


check_basic.example = True
