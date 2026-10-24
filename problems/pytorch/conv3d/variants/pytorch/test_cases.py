import torch
import torch.nn.functional as F


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_matches_torch(solve):
    torch.manual_seed(0)
    x = torch.randn(1, 2, 5, 5, 5)
    for stride, padding in [(1, 0), (1, 1), (2, 0)]:
        w = torch.randn(3, 2, 2, 2, 2)
        b = torch.randn(3)
        got = solve(x, w, b, stride, padding)
        ref = F.conv3d(x, w, b, stride=stride, padding=padding)
        assert torch.allclose(got, ref, atol=1e-5), f"stride={stride} pad={padding}"


def check_edge_example(solve):
    # 手算样例：arange(1..8) (1,1,2,2,2) 全 1 核 → 36
    x = torch.arange(1, 9, dtype=torch.float32).view(1, 1, 2, 2, 2)
    w = torch.ones(1, 1, 2, 2, 2)
    got = solve(x, w, torch.zeros(1), 1, 0)
    assert got.shape == (1, 1, 1, 1, 1)
    assert torch.allclose(got.flatten(), torch.tensor([36.0]), atol=1e-5), got.flatten().tolist()


def check_padding_shape(solve):
    # padding=1 应把输出各维从 1 扩到 3
    x = torch.arange(1, 9, dtype=torch.float32).view(1, 1, 2, 2, 2)
    w = torch.ones(1, 1, 2, 2, 2)
    got = solve(x, w, torch.zeros(1), 1, 1)
    assert got.shape == (1, 1, 3, 3, 3)
    assert torch.allclose(got, F.conv3d(x, w, torch.zeros(1), stride=1, padding=1), atol=1e-5)


def check_no_bias(solve):
    torch.manual_seed(2)
    x = torch.randn(1, 2, 4, 4, 4)
    w = torch.randn(3, 2, 3, 3, 3)
    got = solve(x, w, None, 1, 0)
    assert torch.allclose(got, F.conv3d(x, w, None), atol=1e-5)


check_matches_torch.example = True
