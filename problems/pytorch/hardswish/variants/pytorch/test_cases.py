import torch


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic(solve):
    x = torch.tensor([-4.0, -3.0, 0.0, 3.0, 4.0])
    out = solve(x)
    expected = torch.tensor([0.0, 0.0, 0.0, 3.0, 4.0])
    assert torch.allclose(out, expected, atol=1e-6), f"got {out.tolist()}, expected {expected.tolist()}"


def check_matches_torch(solve):
    torch.manual_seed(42)
    x = torch.randn(64, 32) * 3
    assert torch.allclose(solve(x), torch.nn.functional.hardswish(x), atol=1e-6), "与 F.hardswish 不一致"


def check_mid_slope(solve):
    # x=0 处 relu6(3)/6 = 0.5，输出 0；x=1 处 1*clamp(4)/6=4/6
    x = torch.tensor([1.0])
    out = solve(x)
    assert torch.allclose(out, torch.tensor([4.0 / 6.0]), atol=1e-6), f"中间段错误: got {out.tolist()}"


def check_saturates(solve):
    # |x| 超出 [-3,3] 平台区：x>=3 恒等，x<=-3 归零
    x = torch.tensor([-100.0, 100.0])
    out = solve(x)
    assert torch.allclose(out, torch.tensor([0.0, 100.0]), atol=1e-6), f"饱和错误: got {out.tolist()}"


def check_shape_preserved(solve):
    x = torch.randn(4, 5)
    out = solve(x)
    assert out.shape == x.shape and out.dtype == x.dtype, "形状/dtype 必须保持"


check_basic.example = True
