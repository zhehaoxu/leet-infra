import torch


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic(solve):
    x = torch.tensor([-1.0, 0.0, 1.0])
    out = solve(x, 1.0)
    expected = torch.tensor([-0.63212056, 0.0, 1.0])  # expm1(-1), 0, 1
    assert torch.allclose(out, expected, atol=1e-6), f"got {out.tolist()}, expected {expected.tolist()}"


def check_matches_torch(solve):
    torch.manual_seed(42)
    x = torch.randn(64, 32)
    assert torch.allclose(solve(x, 1.0), torch.nn.functional.elu(x, 1.0), atol=1e-6), "与 F.elu 不一致"
    assert torch.allclose(solve(x, 2.5), torch.nn.functional.elu(x, 2.5), atol=1e-6), "alpha=2.5 不一致"


def check_negative_asymptote(solve):
    # x 很负时 exp(x)->0，输出应趋近 -alpha
    x = torch.tensor([-50.0, -30.0])
    out = solve(x, 2.0)
    assert torch.allclose(out, torch.tensor([-2.0, -2.0]), atol=1e-6), f"负区应饱和到 -alpha, got {out.tolist()}"


def check_small_negative_precision(solve):
    # 近零负值：expm1 保证 -1e-8 量级不被 float32 抵消成 0
    x = torch.tensor([-1e-8])
    out = solve(x, 1.0)
    assert abs(out.item() - (-1e-8)) < 1e-12, f"近零精度丢失: got {out.item()}"


def check_shape_preserved(solve):
    x = torch.randn(2, 3, 4)
    out = solve(x, 1.0)
    assert out.shape == x.shape and out.dtype == x.dtype, "形状/dtype 必须保持"


check_basic.example = True
