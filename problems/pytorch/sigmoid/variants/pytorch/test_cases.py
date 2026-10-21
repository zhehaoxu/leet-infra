import torch


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic(solve):
    x = torch.tensor([-2.0, 0.0, 2.0])
    out = solve(x)
    expected = torch.tensor([0.11920292, 0.5, 0.88079708])
    assert torch.allclose(out, expected, atol=1e-6), f"expected {expected.tolist()}, got {out.tolist()}"


def check_matches_torch_sigmoid(solve):
    torch.manual_seed(42)
    x = torch.randn(64, 32)
    out = solve(x)
    assert out.shape == x.shape, f"形状应保持: expected {x.shape}, got {out.shape}"
    assert torch.allclose(out, torch.sigmoid(x), atol=1e-6), "与 torch.sigmoid 结果不一致"


def check_extreme_no_nan(solve):
    # ±800 会让朴素 exp(-x) 上溢；稳定实现必须给出精确 1.0 / 0.0 且无 NaN
    x = torch.tensor([800.0, 800.0, 0.0, -800.0, -800.0])
    out = solve(x)
    assert torch.isfinite(out).all(), f"输出出现 inf/NaN: {out.tolist()}"
    expected = torch.tensor([1.0, 1.0, 0.5, 0.0, 0.0])
    assert torch.allclose(out, expected), f"极端值结果错误: got {out.tolist()}"


def check_output_range_and_monotonic(solve):
    torch.manual_seed(42)
    x = torch.linspace(-16, 16, 256)  # 避开 fp32 下 sigmoid→1.0 舍入的区间，开区间断言才稳定
    out = solve(x)
    assert (out > 0).all() and (out < 1).all(), "输出应严格落在 (0,1) 开区间"
    assert (out[1:] >= out[:-1]).all(), "sigmoid 应单调递增"


def check_float64(solve):
    torch.manual_seed(42)
    x = torch.randn(32, dtype=torch.float64) * 10
    out = solve(x)
    assert out.dtype == torch.float64, f"dtype 应保持 float64, got {out.dtype}"
    assert torch.allclose(out, torch.sigmoid(x), atol=1e-12), "float64 输入结果不一致"


check_basic.example = True
