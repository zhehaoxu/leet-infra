import torch
import torch.nn.functional as F


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_matches_torch(solve):
    torch.manual_seed(0)
    x = torch.randn(2, 6, 4, 4)
    for size in (3, 5):
        for alpha, beta, k in [(1e-4, 0.75, 1.0), (0.5, 1.0, 2.0)]:
            got = solve(x, size, alpha, beta, k)
            ref = F.local_response_norm(x, size, alpha=alpha, beta=beta, k=k)
            assert torch.allclose(got, ref, atol=1e-5), f"size={size} a={alpha} b={beta} k={k}"


def check_edge_example(solve):
    # 手算样例：x=[1,2,3] (1,3,1,1), size=3, alpha=0.5, beta=1, k=1
    x = torch.tensor([[[[1.0]], [[2.0]], [[3.0]]]])
    got = solve(x, 3, 0.5, 1.0, 1.0).flatten()
    ref = torch.tensor([0.545455, 0.6, 0.947368])
    assert torch.allclose(got, ref, atol=1e-5), got.tolist()


def check_normalization_suppresses(solve):
    # 大激活所在通道被邻通道平方和抑制：输出幅值应严格小于输入幅值
    torch.manual_seed(1)
    x = torch.randn(1, 8, 3, 3).abs() + 0.5
    got = solve(x, 5, 1.0, 1.0, 1.0)
    assert torch.all(got.abs() <= x.abs() + 1e-6)
    assert torch.any(got.abs() < x.abs() - 1e-3), "归一化应显著缩小至少部分通道"


def check_shape_preserved(solve):
    x = torch.randn(3, 4, 5, 5)
    assert solve(x, 3).shape == x.shape


check_matches_torch.example = True
