import torch
import torch.nn.functional as F


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_matches_torch(solve):
    torch.manual_seed(0)
    x = torch.randn(2, 3, 12)
    for stride, padding in [(1, 0), (1, 1), (2, 0), (2, 3)]:
        w = torch.randn(4, 3, 3)
        b = torch.randn(4)
        got = solve(x, w, b, stride, padding)
        ref = F.conv1d(x, w, b, stride=stride, padding=padding)
        assert torch.allclose(got, ref, atol=1e-5), f"stride={stride} pad={padding}"


def check_edge_example(solve):
    # 手算样例：x=[[1,2,3,4,5]], w=[[1,2,3]], bias=0 → [14,20,26]
    x = torch.tensor([[[1.0, 2, 3, 4, 5]]])
    w = torch.tensor([[[1.0, 2, 3]]])
    got = solve(x, w, torch.zeros(1), 1, 0).flatten()
    ref = torch.tensor([14.0, 20.0, 26.0])
    assert torch.allclose(got, ref, atol=1e-5), got.tolist()


def check_no_bias(solve):
    torch.manual_seed(1)
    x = torch.randn(1, 2, 8)
    w = torch.randn(3, 2, 3)
    got = solve(x, w, None, 1, 1)
    ref = F.conv1d(x, w, None, stride=1, padding=1)
    assert torch.allclose(got, ref, atol=1e-5)


def check_multi_channel(solve):
    # 多输入/输出通道：每个输出通道对所有输入通道求和
    torch.manual_seed(2)
    x = torch.randn(2, 5, 16)
    w = torch.randn(7, 5, 4)
    b = torch.randn(7)
    assert torch.allclose(solve(x, w, b, 1, 0), F.conv1d(x, w, b), atol=1e-5)


check_matches_torch.example = True
