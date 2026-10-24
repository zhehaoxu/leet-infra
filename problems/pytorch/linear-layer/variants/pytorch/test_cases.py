import torch
import torch.nn.functional as F


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic(solve):
    x = torch.tensor([[1.0, 2, 3], [4, 5, 6]])
    w = torch.tensor([[1.0, 0, 2], [-1, 3, 0]])
    b = torch.tensor([10.0, -1])
    y = solve(x, w, b)
    exp = [[17.0, 4.0], [26.0, 10.0]]
    assert torch.allclose(y, torch.tensor(exp), atol=1e-5), y.tolist()


def check_no_bias(solve):
    torch.manual_seed(0)
    x = torch.randn(3, 5)
    w = torch.randn(4, 5)
    assert torch.allclose(solve(x, w), F.linear(x, w), atol=1e-4)


def check_batched(solve):
    torch.manual_seed(1)
    x = torch.randn(2, 6, 5)          # 带 batch/序列前置维
    w = torch.randn(7, 5)
    b = torch.randn(7)
    y = solve(x, w, b)
    assert y.shape == (2, 6, 7)
    assert torch.allclose(y, F.linear(x, w, b), atol=1e-4)


def check_weight_convention(solve):
    # weight 是 (out, in)：把 in/out 弄反应触发形状不匹配（这里正向验证形状）
    torch.manual_seed(2)
    x = torch.randn(4, 9)
    w = torch.randn(3, 9)
    assert solve(x, w).shape == (4, 3)


check_basic.example = True
