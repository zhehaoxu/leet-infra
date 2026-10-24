import torch
import torch.nn.functional as F


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic(solve):
    x = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    y = solve(x, 2)
    exp = torch.tensor([[[[5, 7], [13, 15]]]], dtype=torch.float32)
    assert torch.equal(y, exp), f"expected {exp.tolist()}, got {y.tolist()}"


def check_matches_F(solve):
    torch.manual_seed(0)
    x = torch.randn(2, 3, 5, 6)
    y = solve(x, (2, 3))
    ref = F.adaptive_max_pool2d(x, (2, 3))
    assert y.shape == (2, 3, 2, 3)
    assert torch.allclose(y, ref, atol=1e-6), "应与 F.adaptive_max_pool2d 一致"


def check_global_max(solve):
    # output_size=1 时即逐通道全局最大值
    torch.manual_seed(1)
    x = torch.randn(4, 3, 7, 5)
    y = solve(x, 1)
    ref = F.adaptive_max_pool2d(x, 1)
    assert y.shape == (4, 3, 1, 1)
    assert torch.allclose(y, ref, atol=1e-6)


def check_max_not_mean(solve):
    # 与 avg 区分：负值段里 max 取的是最不负的那个，而非均值
    x = torch.full((1, 1, 2, 2), -3.0)
    x[0, 0, 0, 0] = -1.0
    y = solve(x, 1)
    assert y.item() == -1.0, "max 池化应取段内最大值"


check_basic.example = True
