import math

import torch


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic_exact(solve):
    x = torch.tensor([-1.0, 0.0, 1.0])
    out = solve(x, approximate="none")
    expected = torch.tensor([-0.15865526, 0.0, 0.84134474])
    assert torch.allclose(out, expected, atol=1e-5), f"expected {expected.tolist()}, got {out.tolist()}"


def check_exact_matches_torch(solve):
    import torch.nn.functional as F

    torch.manual_seed(42)
    x = torch.randn(64, 32)
    out = solve(x, approximate="none")
    assert out.shape == x.shape, "形状应保持"
    assert torch.allclose(out, F.gelu(x, approximate="none"), atol=1e-6), "精确式与 F.gelu 不一致"


def check_tanh_matches_torch(solve):
    import torch.nn.functional as F

    torch.manual_seed(42)
    x = torch.randn(64, 32)
    out = solve(x, approximate="tanh")
    assert torch.allclose(out, F.gelu(x, approximate="tanh"), atol=1e-6), "tanh 近似与 F.gelu 不一致"


def check_two_modes_close(solve):
    # tanh 近似与精确式偏差很小（|x|<=4 内约 <2e-3），但不应逐位相等
    torch.manual_seed(42)
    x = torch.randn(256).clamp(-4, 4)
    a = solve(x, approximate="none")
    b = solve(x, approximate="tanh")
    assert torch.allclose(a, b, atol=2e-3), "两种模式偏差应 < 2e-3"
    assert not torch.equal(a, b), "两种模式不应输出完全相同的值"


def check_invalid_approximate(solve):
    try:
        solve(torch.randn(4), approximate="bad")
    except ValueError:
        return
    raise AssertionError("非法 approximate 参数应抛出 ValueError")


def check_zero_and_large_negative(solve):
    x = torch.tensor([0.0, -10.0, 10.0])
    out = solve(x, approximate="none")
    assert out[0].item() == 0.0, "gelu(0) 应恰好为 0"
    assert torch.isfinite(out).all(), "大 |x| 下不应出现 NaN/inf"
    assert out[1].abs().item() < 1e-3, "x=-10 应几乎为 0"
    assert abs(out[2].item() - 10.0) < 1e-3, "x=10 应几乎等于 x"


check_basic_exact.example = True
