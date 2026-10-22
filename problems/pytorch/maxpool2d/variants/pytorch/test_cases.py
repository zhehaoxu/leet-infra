import torch
import torch.nn.functional as F


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic(solve):
    x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0],
                        [5.0, 6.0, 7.0, 8.0],
                        [9.0, 10.0, 11.0, 12.0],
                        [13.0, 14.0, 15.0, 16.0]]]])
    y = solve(x, 2)
    expected = torch.tensor([[[[6.0, 8.0], [14.0, 16.0]]]])
    assert torch.allclose(y, expected), f"got {y.tolist()}, expected {expected.tolist()}"


def check_matches_torch(solve):
    torch.manual_seed(42)
    x = torch.randn(2, 3, 7, 5)
    for k, s, p in [(2, 2, 0), (3, 1, 1), (3, 2, 1), (2, 3, 0), ((2, 3), (1, 2), (1, 0))]:
        y = solve(x, k, s, p)
        exp = F.max_pool2d(x, k, s, p)
        assert torch.allclose(y, exp), f"k={k} s={s} p={p} 与 F.max_pool2d 不一致"


def check_default_stride(solve):
    torch.manual_seed(0)
    x = torch.randn(2, 2, 6, 6)
    y = solve(x, 3)
    assert torch.allclose(y, F.max_pool2d(x, 3, 3), atol=1e-6), "stride 缺省应等于 kernel_size"


def check_negative_padding(solve):
    # 全负输入 + padding=1：填 0 的错误实现会在边界窗口给出 0
    x = -torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4) - 1.0
    y = solve(x, 3, 1, 1)
    exp = F.max_pool2d(x, 3, 1, 1)
    assert torch.allclose(y, exp, atol=1e-6), f"padding 必须填 -inf；got {y.tolist()}"
    assert (y < 0).all(), "全负输入的输出不应出现 0"


def check_shape_formula(solve):
    torch.manual_seed(1)
    x = torch.randn(4, 8, 17, 13)
    y = solve(x, 3, 2, 1)
    assert y.shape == (4, 8, 9, 7), f"输出形状错误 {tuple(y.shape)}"


def check_float64(solve):
    x = torch.randn(1, 2, 5, 5, dtype=torch.float64)
    y = solve(x, 2, 1, 0)
    assert y.dtype == torch.float64
    assert torch.allclose(y, F.max_pool2d(x, 2, 1, 0), atol=1e-12)


check_basic.example = True
