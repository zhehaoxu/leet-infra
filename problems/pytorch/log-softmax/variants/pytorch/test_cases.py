import torch
import torch.nn.functional as F


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic(solve):
    x = torch.tensor([[1.0, 2, 3], [3, 1, 0]])
    y = solve(x, -1)
    exp = torch.tensor([[-2.4076, -1.4076, -0.4076],
                        [-0.1698, -2.1698, -3.1698]])
    assert torch.allclose(y, exp, atol=1e-3), y.tolist()


def check_matches_F(solve):
    torch.manual_seed(0)
    x = torch.randn(5, 9)
    assert torch.allclose(solve(x, -1), F.log_softmax(x, -1), atol=1e-6)


def check_exp_sums_to_one(solve):
    torch.manual_seed(1)
    x = torch.randn(4, 6)
    assert torch.allclose(solve(x, -1).exp().sum(-1), torch.ones(4), atol=1e-5)


def check_no_overflow(solve):
    # logits 很大时朴素 log(softmax) 会溢出，稳定版须保持有限
    x = torch.tensor([[1000.0, 1001, 999]])
    y = solve(x, -1)
    assert torch.isfinite(y).all(), "大 logits 不应产生 inf/nan"
    assert torch.allclose(y, F.log_softmax(x, -1), atol=1e-6)


def check_other_dim(solve):
    torch.manual_seed(2)
    x = torch.randn(3, 4, 5)
    assert torch.allclose(solve(x, 1), F.log_softmax(x, 1), atol=1e-6)


check_basic.example = True
