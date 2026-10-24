import torch
import torch.nn.functional as F


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic(solve):
    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0, 7.0])
    out = solve(x)
    expected = torch.tensor([-1.52017, -0.69176, 0.0, 0.52535, 2.1014, 7.35491])
    assert torch.allclose(out, expected, atol=1e-4), f"got {out.tolist()}"


def check_zero_is_zero(solve):
    out = solve(torch.tensor([0.0]))
    assert abs(out.item()) < 1e-12, "SELU(0) 应为 0（连续）"


def check_positive_scale(solve):
    lam = 1.0507009873554804977367086399070
    out = solve(torch.tensor([3.0]))
    assert abs(out.item() - lam * 3.0) < 1e-5, "x>0 时应为 λ*x"


def check_matches_F(solve):
    torch.manual_seed(42)
    x = torch.randn(64, 32)
    assert torch.allclose(solve(x), F.selu(x), atol=1e-6), "应与 F.selu 一致"


def check_float64(solve):
    torch.manual_seed(42)
    x = torch.randn(16, dtype=torch.float64)
    out = solve(x)
    assert out.dtype == torch.float64, f"dtype 应保持, got {out.dtype}"
    assert torch.allclose(out, F.selu(x), atol=1e-9)


check_basic.example = True
