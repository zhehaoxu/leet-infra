import torch
import torch.nn.functional as F


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic(solve):
    x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0],
                        [5.0, 6.0, 7.0, 8.0],
                        [9.0, 10.0, 11.0, 12.0],
                        [13.0, 14.0, 15.0, 16.0]]]])
    y = solve(x, 2)
    expected = torch.tensor([[[[3.5, 5.5], [11.5, 13.5]]]])
    assert torch.allclose(y, expected), f"got {y.tolist()}, expected {expected.tolist()}"


def check_matches_torch(solve):
    torch.manual_seed(42)
    x = torch.randn(2, 3, 7, 5)
    for k, s, p, cip in [(2, 2, 0, True), (3, 1, 1, True), (3, 2, 1, True),
                         (3, 1, 1, False), (2, 2, 1, False), ((2, 3), (1, 2), (1, 1), False)]:
        y = solve(x, k, s, p, cip)
        exp = F.avg_pool2d(x, k, s, p, count_include_pad=cip)
        assert torch.allclose(y, exp, atol=1e-6), f"k={k} s={s} p={p} cip={cip} 与 F.avg_pool2d 不一致"


def check_count_include_pad(solve):
    # 经典坑：同一个全 1 输入，padding=1 时两种分母口径结果不同
    x = torch.ones(1, 1, 4, 4)
    y1 = solve(x, 3, 1, 1, count_include_pad=True)
    y2 = solve(x, 3, 1, 1, count_include_pad=False)
    assert torch.allclose(y1[..., 0, 0], torch.full(y1[..., 0, 0].shape, 4 / 9)), "角点应除以窗口面积 9"
    assert torch.allclose(y2[..., 0, 0], torch.ones_like(y2[..., 0, 0])), "角点应除以有效数 4 得 1"
    assert not torch.allclose(y1, y2), "两种口径必须不同"


def check_default_stride(solve):
    torch.manual_seed(0)
    x = torch.randn(2, 2, 6, 6)
    y = solve(x, 3)
    assert torch.allclose(y, F.avg_pool2d(x, 3, 3), atol=1e-6), "stride 缺省应等于 kernel_size"


def check_ceil_shape(solve):
    torch.manual_seed(1)
    x = torch.randn(4, 8, 17, 13)
    y = solve(x, 3, 2, 1)
    assert y.shape == (4, 8, 9, 7), f"输出形状错误 {tuple(y.shape)}"


def check_float64(solve):
    x = torch.randn(1, 2, 5, 5, dtype=torch.float64)
    y = solve(x, 2, 1, 1, False)
    assert y.dtype == torch.float64
    assert torch.allclose(y, F.avg_pool2d(x, 2, 1, 1, count_include_pad=False), atol=1e-12)


check_basic.example = True
