import torch
import torch.nn.functional as F


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_matches_torch_grouped(solve):
    torch.manual_seed(0)
    x = torch.randn(2, 4, 6, 6)
    for g, c in [(1, 8), (2, 8), (4, 4)]:
        w = torch.randn(c, 4 // g, 3, 3)
        b = torch.randn(c)
        got = solve(x, w, b, g)
        assert torch.allclose(got, F.conv2d(x, w, b, groups=g), atol=1e-5), f"g={g}"


def check_group_locality(solve):
    # 扰动某组的输入通道不应改变另一组的输出通道
    torch.manual_seed(1)
    x = torch.randn(1, 4, 5, 5)
    w = torch.randn(4, 2, 3, 3)
    b = torch.zeros(4)
    y = solve(x, w, b, 2)
    x2 = x.clone()
    x2[:, 0] += 10.0  # 通道 0 属 group0
    y2 = solve(x2, w, b, 2)
    assert torch.allclose(y[:, 2:], y2[:, 2:], atol=1e-6), "group1 输出不应受 group0 输入影响"
    assert not torch.allclose(y[:, 0], y2[:, 0], atol=1e-4), "本组输出应受本组输入影响"


def check_single_group_equals_conv(solve):
    torch.manual_seed(2)
    x = torch.randn(1, 3, 5, 5)
    w = torch.randn(5, 3, 3, 3)
    b = torch.randn(5)
    assert torch.allclose(solve(x, w, b, 1), F.conv2d(x, w, b), atol=1e-5)


def check_depthwise_case(solve):
    # groups == Cin 即深度卷积：每个输入通道各用自己的核
    torch.manual_seed(3)
    x = torch.randn(1, 4, 5, 5)
    w = torch.randn(4, 1, 3, 3)
    b = torch.randn(4)
    assert torch.allclose(solve(x, w, b, 4), F.conv2d(x, w, b, groups=4), atol=1e-5)


check_matches_torch_grouped.example = True
