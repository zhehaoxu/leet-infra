import torch


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic(solve):
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    out = solve(x, 0.01)
    expected = torch.tensor([-0.02, -0.01, 0.0, 1.0, 2.0])
    assert torch.allclose(out, expected, atol=1e-7), f"got {out.tolist()}, expected {expected.tolist()}"


def check_zero_is_zero(solve):
    x = torch.tensor([0.0])
    assert solve(x, 0.1).item() == 0.0, "0 应输出 0"


def check_matches_torch(solve):
    torch.manual_seed(42)
    x = torch.randn(64, 32)
    assert torch.allclose(solve(x, 0.2), torch.nn.functional.leaky_relu(x, 0.2), atol=1e-6), \
        "与 F.leaky_relu(0.2) 不一致"


def check_slope_one_is_identity(solve):
    # negative_slope=1 时退化为恒等
    torch.manual_seed(1)
    x = torch.randn(16)
    assert torch.equal(solve(x, 1.0), x), "slope=1 应原样返回"


def check_shape_preserved(solve):
    x = torch.randn(3, 5, 7)
    out = solve(x, 0.01)
    assert out.shape == x.shape and out.dtype == x.dtype, "形状/dtype 必须保持"


check_basic.example = True
