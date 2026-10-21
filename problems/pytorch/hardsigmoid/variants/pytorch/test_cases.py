import torch


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic(solve):
    x = torch.tensor([-4.0, -3.0, 0.0, 3.0, 4.0])
    out = solve(x)
    expected = torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0])
    assert torch.allclose(out, expected, atol=1e-6), f"got {out.tolist()}, expected {expected.tolist()}"


def check_matches_torch(solve):
    torch.manual_seed(42)
    x = torch.randn(64, 32) * 3
    assert torch.allclose(solve(x), torch.nn.functional.hardsigmoid(x), atol=1e-6), "与 F.hardsigmoid 不一致"


def check_range_unit(solve):
    torch.manual_seed(0)
    x = torch.randn(1000) * 10
    out = solve(x)
    assert (out >= 0).all() and (out <= 1).all(), "HardSigmoid 必须落在 [0,1]"


def check_linear_ramp(solve):
    # 斜坡段：x=-1 -> clamp(2)/6 = 1/3
    x = torch.tensor([-1.0])
    assert torch.allclose(solve(x), torch.tensor([1.0 / 3.0]), atol=1e-6), "斜坡段错误"


def check_shape_preserved(solve):
    x = torch.randn(2, 6)
    out = solve(x)
    assert out.shape == x.shape and out.dtype == x.dtype, "形状/dtype 必须保持"


check_basic.example = True
