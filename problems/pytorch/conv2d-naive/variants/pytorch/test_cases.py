import torch
import torch.nn.functional as F


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic(solve):
    # 单通道 3x3 输入、恒等核对用 2x2 全 1 核：输出 = 2x2 窗口求和
    x = torch.tensor([[[[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0]]]])
    w = torch.ones(1, 1, 2, 2)
    y = solve(x, w)
    expected = torch.tensor([[[[12.0, 16.0], [24.0, 28.0]]]])
    assert torch.allclose(y, expected), f"got {y.tolist()}, expected {expected.tolist()}"


def check_matches_torch(solve):
    torch.manual_seed(42)
    x = torch.randn(2, 3, 6, 5)
    w = torch.randn(4, 3, 3, 3)
    b = torch.randn(4)
    for s, p in [(1, 0), (1, 1), (2, 0), (2, 1)]:
        y = solve(x, w, b, s, p)
        exp = F.conv2d(x, w, b, s, p)
        assert torch.allclose(y, exp, atol=1e-4), f"s={s} p={p} 与 F.conv2d 不一致"


def check_1x1_conv(solve):
    # 1x1 卷积 = 逐位置线性映射（检验通道维处理）
    torch.manual_seed(0)
    x = torch.randn(1, 2, 3, 3)
    w = torch.randn(3, 2, 1, 1)
    y = solve(x, w)
    assert torch.allclose(y, F.conv2d(x, w), atol=1e-4), "1x1 卷积不一致"


def check_no_bias_zero(solve):
    # 全零输入任意核：输出必为 0（bias=None）
    x = torch.zeros(1, 2, 4, 4)
    w = torch.randn(3, 2, 3, 3)
    y = solve(x, w, None, 1, 1)
    assert torch.allclose(y, torch.zeros(1, 3, 4, 4), atol=1e-7), "零输入应得零输出"


def check_shape_formula(solve):
    torch.manual_seed(1)
    x = torch.randn(1, 4, 9, 7)
    w = torch.randn(6, 4, 3, 5)
    y = solve(x, w, None, 2, 1)
    assert y.shape == (1, 6, 5, 3), f"输出形状错误 {tuple(y.shape)}"  # floor((9+2-3)/2)+1=5, floor((7+2-5)/2)+1=3


def check_float64(solve):
    x = torch.randn(1, 2, 5, 5, dtype=torch.float64)
    w = torch.randn(3, 2, 3, 3, dtype=torch.float64)
    y = solve(x, w)
    assert y.dtype == torch.float64
    assert torch.allclose(y, F.conv2d(x, w), atol=1e-10)


check_basic.example = True
