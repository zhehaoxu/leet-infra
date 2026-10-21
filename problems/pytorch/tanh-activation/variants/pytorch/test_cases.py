import torch


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic(solve):
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    out = solve(x)
    expected = torch.tanh(x)
    assert torch.allclose(out, expected, atol=1e-6), f"got {out.tolist()}, expected {expected.tolist()}"


def check_matches_torch(solve):
    torch.manual_seed(42)
    x = torch.randn(64, 32)
    assert torch.allclose(solve(x), torch.tanh(x), atol=1e-6), "与 torch.tanh 不一致"


def check_extreme_saturation(solve):
    # exp(2*800) 上溢；正确实现应饱和到 ±1 而非 NaN
    x = torch.tensor([-800.0, -20.0, 20.0, 800.0])
    out = solve(x)
    assert torch.isfinite(out).all(), f"出现 NaN/inf: {out.tolist()}"
    assert torch.allclose(out, torch.tanh(x), atol=1e-6), "极值应饱和到 ±1"


def check_odd_symmetry(solve):
    torch.manual_seed(0)
    x = torch.randn(32)
    assert torch.allclose(solve(-x), -solve(x), atol=1e-6), "tanh 应为奇函数"


def check_float64(solve):
    x = torch.randn(16, dtype=torch.float64)
    out = solve(x)
    assert out.dtype == torch.float64, f"dtype 应保持, got {out.dtype}"
    assert torch.allclose(out, torch.tanh(x), atol=1e-12), "float64 不一致"


check_basic.example = True
