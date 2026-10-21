import torch


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic(solve):
    x = torch.tensor([-2.0, 0.0, 2.0])
    out = solve(x, 1.0, 20.0)
    expected = torch.tensor([0.12692805, 0.69314718, 2.12692801])
    assert torch.allclose(out, expected, atol=1e-6), f"got {out.tolist()}, expected {expected.tolist()}"


def check_matches_torch(solve):
    torch.manual_seed(42)
    x = torch.randn(64, 32)
    assert torch.allclose(solve(x, 1.0, 20.0), torch.nn.functional.softplus(x), atol=1e-6), "与 F.softplus 不一致"
    assert torch.allclose(solve(x, 2.0, 20.0), torch.nn.functional.softplus(x, beta=2.0), atol=1e-6), "beta=2 不一致"


def check_large_no_overflow(solve):
    # exp(800) 上溢；beta*x>threshold 应走线性分支返回 x
    x = torch.tensor([800.0, 100.0])
    out = solve(x, 1.0, 20.0)
    assert torch.isfinite(out).all(), f"出现 inf/NaN: {out.tolist()}"
    assert torch.allclose(out, x, atol=1e-6), "大值应逼近 x"


def check_softplus_of_zero(solve):
    out = solve(torch.tensor([0.0]), 1.0, 20.0)
    assert torch.allclose(out, torch.tensor([0.69314718]), atol=1e-6), "softplus(0)=ln2"


def check_shape_preserved(solve):
    x = torch.randn(3, 4)
    out = solve(x, 1.0, 20.0)
    assert out.shape == x.shape and out.dtype == x.dtype, "形状/dtype 必须保持"


check_basic.example = True
