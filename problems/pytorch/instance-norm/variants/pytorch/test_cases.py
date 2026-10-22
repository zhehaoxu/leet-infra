import torch
import torch.nn.functional as F


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic(solve):
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])  # (N=2, C=2, L=2)
    y = solve(x)
    exp = F.instance_norm(x, None, None, None, None, True, 0.1, 1e-5)
    assert torch.allclose(y, exp, atol=1e-6), f"got {y.tolist()}, expected {exp.tolist()}"


def check_matches_torch(solve):
    torch.manual_seed(42)
    x = torch.randn(8, 4, 6, 6)
    w = torch.randn(4)
    b = torch.randn(4)
    y = solve(x, w, b)
    assert torch.allclose(y, F.instance_norm(x, None, None, w, b, True, 0.1, 1e-5), atol=1e-5), "与 F.instance_norm 不一致"


def check_per_sample_independence(solve):
    # 每个样本独立归一化：改变某一样本不影响其他样本的输出
    torch.manual_seed(0)
    x = torch.randn(4, 3, 8, 8)
    y1 = solve(x)
    x2 = x.clone()
    x2[1] = torch.randn(3, 8, 8) * 100
    y2 = solve(x2)
    for i in (0, 2, 3):
        assert torch.allclose(y1[i], y2[i], atol=1e-6), f"样本 {i} 不应受其他样本影响"
    assert not torch.allclose(y1[1], y2[1], atol=1e-3), "被改样本自身应变化"


def check_no_affine(solve):
    # weight/bias 为 None 时输出均值≈0、方差≈1（逐样本逐通道）
    torch.manual_seed(1)
    x = torch.randn(2, 3, 16, 16) * 5 + 42
    y = solve(x)
    stat_dims = (2, 3)
    assert torch.allclose(y.mean(stat_dims), torch.zeros(2, 3), atol=1e-5), "均值应为 0"
    assert torch.allclose(y.var(stat_dims, unbiased=False), torch.ones(2, 3), rtol=1e-3), "方差应≈1"


def check_const_channel(solve):
    # 某通道为常数：var=0，eps 保护不得 NaN
    x = torch.randn(1, 2, 4, 4)
    x[0, 1] = 7.0
    y = solve(x)
    assert torch.isfinite(y).all(), "常数通道未加 eps 导致 NaN"


check_basic.example = True
