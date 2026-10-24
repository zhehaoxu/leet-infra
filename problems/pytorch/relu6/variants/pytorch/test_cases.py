import torch
import torch.nn.functional as F


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic(solve):
    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0, 7.0])
    out = solve(x)
    expected = torch.tensor([0.0, 0.0, 0.0, 0.5, 2.0, 6.0])
    assert torch.allclose(out, expected), f"got {out.tolist()}"


def check_upper_bound(solve):
    x = torch.tensor([100.0, 6.0, 6.5, 5.999])
    out = solve(x)
    assert out.max().item() <= 6.0, "输出上界应为 6"
    assert torch.allclose(out, torch.tensor([6.0, 6.0, 6.0, 5.999]))


def check_matches_F(solve):
    torch.manual_seed(42)
    x = torch.randn(64, 32) * 5
    assert torch.allclose(solve(x), F.relu6(x)), "应与 F.relu6 一致"


def check_float64(solve):
    torch.manual_seed(42)
    x = torch.randn(16, dtype=torch.float64)
    out = solve(x)
    assert out.dtype == torch.float64, f"dtype 应保持, got {out.dtype}"
    assert torch.allclose(out, F.relu6(x))


check_basic.example = True
