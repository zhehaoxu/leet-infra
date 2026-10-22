import torch
import torch.nn.functional as F


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic(solve):
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


def check_matches_naive_loop(solve):
    # 与显式三重循环参考实现自对拍（im2col 的展开顺序易错：通道必须在行内连续）
    torch.manual_seed(7)
    x = torch.randn(1, 2, 5, 6)
    w = torch.randn(3, 2, 3, 2)
    n, cin, h, wd = x.shape
    cout, _, kh, kw = w.shape
    y_loop = torch.zeros(n, cout, h - kh + 1, wd - kw + 1)
    for i in range(h - kh + 1):
        for j in range(wd - kw + 1):
            region = x[:, :, i:i + kh, j:j + kw]
            for o in range(cout):
                y_loop[:, o, i, j] = (region * w[o]).sum((1, 2, 3))
    y = solve(x, w)
    assert torch.allclose(y, y_loop, atol=1e-4), "im2col 展开顺序/乘累加错误"


def check_1x1_is_matmul(solve):
    # 1x1 核：应与逐位置矩阵乘一致（检验通道维处理）
    torch.manual_seed(3)
    x = torch.randn(2, 5, 4, 4)
    w = torch.randn(8, 5, 1, 1)
    y = solve(x, w)
    exp = F.conv2d(x, w)
    assert torch.allclose(y, exp, atol=1e-4), "1x1 卷积不一致"


def check_shape_and_float64(solve):
    x = torch.randn(1, 4, 9, 7, dtype=torch.float64)
    w = torch.randn(6, 4, 3, 5, dtype=torch.float64)
    y = solve(x, w, None, 2, 1)
    assert y.shape == (1, 6, 5, 3), f"输出形状错误 {tuple(y.shape)}"
    assert y.dtype == torch.float64
    assert torch.allclose(y, F.conv2d(x, w, None, 2, 1), atol=1e-10)


check_basic.example = True
