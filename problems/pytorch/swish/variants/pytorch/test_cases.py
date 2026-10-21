import torch


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic(solve):
    x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
    out = solve(x)
    expected = torch.tensor([-0.14228, -0.26894, 0.0, 0.73106, 2.85772])
    assert torch.allclose(out, expected, atol=1e-4), f"expected {expected.tolist()}, got {out.tolist()}"


def check_matches_torch_silu(solve):
    import torch.nn.functional as F

    torch.manual_seed(42)
    x = torch.randn(64, 32)
    out = solve(x)
    assert out.shape == x.shape, f"形状应保持: expected {x.shape}, got {out.shape}"
    assert torch.allclose(out, F.silu(x), atol=1e-6), "与 F.silu 结果不一致"


def check_negative_dip(solve):
    # swish 在 x≈-1.28 处取负向最小值约 -0.278，这是它与 ReLU 的本质区别
    out = solve(torch.tensor([-1.28]))
    v = out.item()
    assert -0.28 < v < -0.27, f"x=-1.28 处应出现特征性下凹，got {v}"


def check_extreme_no_nan(solve):
    x = torch.tensor([-1000.0, -800.0, 0.0, 800.0, 1000.0])
    out = solve(x)
    assert torch.isfinite(out).all(), f"输出出现 inf/NaN: {out.tolist()}"
    assert abs(out[0].item()) < 1e-5, "大负数应趋近 0"
    assert out[-1].item() > 999.0, "大正数应近似等于 x 本身"


def check_float64(solve):
    torch.manual_seed(42)
    x = torch.randn(32, dtype=torch.float64)
    out = solve(x)
    assert out.dtype == torch.float64, f"dtype 应保持 float64, got {out.dtype}"
    assert torch.allclose(out, x * torch.sigmoid(x), atol=1e-12), "float64 输入结果不一致"


check_basic.example = True
