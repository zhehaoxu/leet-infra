import torch
import torch.nn.functional as F


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic(solve):
    # (N=1, C=4, H=1, W=4)，G=2：通道(0,1)与(2,3)各自成组统计
    x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]], [[5.0, 6.0, 7.0, 8.0]],
                        [[9.0, 10.0, 11.0, 12.0]], [[13.0, 14.0, 15.0, 16.0]]]])
    w = torch.ones(4)
    b = torch.zeros(4)
    y = solve(x, w, b, num_groups=2)
    exp = F.group_norm(x, 2, w, b, 1e-5)
    assert torch.allclose(y, exp, atol=1e-6), f"got {y.tolist()}, expected {exp.tolist()}"


def check_matches_torch(solve):
    torch.manual_seed(42)
    x = torch.randn(8, 6, 5, 5)
    w = torch.randn(6)
    b = torch.randn(6)
    for g in (1, 2, 3, 6):
        y = solve(x, w, b, num_groups=g)
        assert torch.allclose(y, F.group_norm(x, g, w, b, 1e-5), atol=1e-5), f"G={g} 与 F.group_norm 不一致"


def check_group_boundary(solve):
    # G=1 退化为整样本统计；G=C 退化为 InstanceNorm —— 两端行为必须不同
    torch.manual_seed(0)
    x = torch.randn(2, 4, 3, 3)
    ones = torch.ones(4)
    zeros = torch.zeros(4)
    y_g1 = solve(x, ones, zeros, num_groups=1)
    y_g4 = solve(x, ones, zeros, num_groups=4)
    assert not torch.allclose(y_g1, y_g4, atol=1e-3), "分组不同结果必须不同"
    # G=1：每样本全通道均值≈0
    per_sample_mean = y_g1.reshape(2, -1).mean(dim=1)
    assert torch.allclose(per_sample_mean, torch.zeros(2), atol=1e-5), "G=1 应整样本归一化"
    # G=C：每样本每通道均值≈0
    per_ch_mean = y_g4.reshape(2, 4, -1).mean(dim=2)
    assert torch.allclose(per_ch_mean, torch.zeros(2, 4), atol=1e-5), "G=C 应逐通道(实例)归一化"


def check_2d_input(solve):
    # (N, C) 无空间维也应工作
    torch.manual_seed(1)
    x = torch.randn(4, 8)
    w = torch.randn(8)
    b = torch.randn(8)
    y = solve(x, w, b, num_groups=4)
    assert torch.allclose(y, F.group_norm(x, 4, w, b, 1e-5), atol=1e-5)


def check_float64(solve):
    x = torch.randn(3, 6, 4, dtype=torch.float64)
    w = torch.randn(6, dtype=torch.float64)
    b = torch.randn(6, dtype=torch.float64)
    y = solve(x, w, b, num_groups=3)
    assert y.dtype == torch.float64
    assert torch.allclose(y, F.group_norm(x, 3, w, b, 1e-5), atol=1e-10)


check_basic.example = True
