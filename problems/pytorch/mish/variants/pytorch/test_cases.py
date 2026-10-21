import torch


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic(solve):
    x = torch.tensor([-1.0, 0.0, 1.0])
    out = solve(x)
    expected = torch.tensor([-0.30340144, 0.0, 0.86509285])
    assert torch.allclose(out, expected, atol=1e-6), f"got {out.tolist()}, expected {expected.tolist()}"


def check_matches_torch(solve):
    torch.manual_seed(42)
    x = torch.randn(64, 32)
    assert torch.allclose(solve(x), torch.nn.functional.mish(x), atol=1e-6), "与 F.mish 不一致"


def check_zero_is_zero(solve):
    assert abs(solve(torch.tensor([0.0])).item()) < 1e-7, "mish(0)=0"


def check_large_positive_identity(solve):
    # x 很大时 softplus≈x，tanh≈1，mish≈x
    x = torch.tensor([50.0, 800.0])
    out = solve(x)
    assert torch.isfinite(out).all(), f"大值溢出: {out.tolist()}"
    assert torch.allclose(out, x, atol=1e-4), f"x 很大时 mish≈x, got {out.tolist()}"


def check_large_negative_zero(solve):
    # x 很负时 softplus→0，tanh→0，mish→0
    x = torch.tensor([-50.0])
    out = solve(x)
    assert abs(out.item()) < 1e-6, f"x 很负时 mish→0, got {out.item()}"


check_basic.example = True
