import torch


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic(solve):
    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    out = solve(x)
    expected = torch.tensor([0.0, 0.0, 0.0, 0.5, 2.0])
    assert torch.allclose(out, expected), f"expected {expected.tolist()}, got {out.tolist()}"


def check_all_negative(solve):
    torch.manual_seed(42)
    x = torch.randn(8, 16) - 5.0
    out = solve(x)
    assert torch.equal(out, torch.zeros_like(out)), "全负输入应输出全零"


def check_matches_torch_relu(solve):
    torch.manual_seed(42)
    x = torch.randn(64, 32)
    out = solve(x)
    assert out.shape == x.shape, f"形状应保持: expected {x.shape}, got {out.shape}"
    assert torch.allclose(out, torch.relu(x)), "与 torch.relu 结果不一致"


def check_float64(solve):
    torch.manual_seed(42)
    x = torch.randn(16, dtype=torch.float64)
    out = solve(x)
    assert out.dtype == torch.float64, f"dtype 应保持 float64, got {out.dtype}"
    assert torch.allclose(out, torch.relu(x)), "float64 输入结果不一致"


check_basic.example = True
