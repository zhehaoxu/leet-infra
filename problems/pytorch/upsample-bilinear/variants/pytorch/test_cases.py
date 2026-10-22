import torch
import torch.nn.functional as F


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic(solve):
    x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    y = solve(x, (4, 4))
    exp = F.interpolate(x, size=(4, 4), mode="bilinear", align_corners=False)
    assert torch.allclose(y, exp, atol=1e-6), f"got {y.tolist()}, expected {exp.tolist()}"


def check_matches_torch(solve):
    torch.manual_seed(42)
    x = torch.randn(2, 3, 5, 7)
    for out, ac in [((10, 14), False), ((10, 14), True), ((3, 4), False), ((3, 4), True), ((5, 7), False)]:
        y = solve(x, out, ac)
        exp = F.interpolate(x, size=out, mode="bilinear", align_corners=ac)
        assert torch.allclose(y, exp, atol=1e-5), f"out={out} align_corners={ac} 与 F.interpolate 不一致"


def check_align_corners_difference(solve):
    # 经典坑：两种角点对齐口径在边界采样位置上不同
    x = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]])
    y_false = solve(x, (4, 4), align_corners=False)
    y_true = solve(x, (4, 4), align_corners=True)
    assert not torch.allclose(y_false, y_true, atol=1e-6), "align_corners 两种取值必须不同"
    assert torch.allclose(y_true[..., 0, 0], x[..., 0, 0].reshape(1, 1, 1, 1), atol=1e-6), "True 时角点应保持原值"


def check_constant_preserved(solve):
    # 双线性是凸组合：常数图上采样后仍是同一常数
    x = torch.full((1, 2, 3, 3), 7.0)
    y = solve(x, (6, 9))
    assert torch.allclose(y, torch.full((1, 2, 6, 9), 7.0), atol=1e-6), "常数应被保持"


def check_linear_ramp(solve):
    # 沿 y 方向的线性梯度，1D 线性插值语义：x=[0,1] -> [0, 1/3, 2/3, 1]（align_corners=True）
    x = torch.tensor([[[[0.0, 0.0], [1.0, 1.0]]]])
    y = solve(x, (4, 2), align_corners=True)
    exp = F.interpolate(x, size=(4, 2), mode="bilinear", align_corners=True)
    assert torch.allclose(y, exp[..., :], atol=1e-6)
    assert torch.allclose(y[0, 0, :, 0], torch.tensor([0.0, 1 / 3, 2 / 3, 1.0]), atol=1e-6), "线性渐变插值错误"


def check_float64(solve):
    x = torch.randn(1, 1, 4, 4, dtype=torch.float64)
    y = solve(x, (8, 8))
    assert y.dtype == torch.float64
    assert torch.allclose(y, F.interpolate(x, size=(8, 8), mode="bilinear", align_corners=False), atol=1e-12)


check_basic.example = True
