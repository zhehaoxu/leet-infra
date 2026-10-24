import torch
import torch.nn.functional as F


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_mean(solve):
    pred = torch.tensor([2.5, 0.0, 2.0, 8.0])
    target = torch.tensor([3.0, -1.0, 2.0, 7.0])
    out = solve(pred, target)
    assert abs(out.item() - 0.5625) < 1e-6, f"got {out.item()}"


def check_none(solve):
    pred = torch.tensor([2.5, 0.0, 2.0, 8.0])
    target = torch.tensor([3.0, -1.0, 2.0, 7.0])
    out = solve(pred, target, "none")
    expected = torch.tensor([0.25, 1.0, 0.0, 1.0])
    assert torch.allclose(out, expected, atol=1e-6), f"got {out.tolist()}"


def check_sum(solve):
    pred = torch.tensor([2.5, 0.0, 2.0, 8.0])
    target = torch.tensor([3.0, -1.0, 2.0, 7.0])
    assert abs(solve(pred, target, "sum").item() - 2.25) < 1e-6


def check_matches_F(solve):
    torch.manual_seed(42)
    p = torch.randn(64, 8)
    t = torch.randn(64, 8)
    for red in ("mean", "sum", "none"):
        assert torch.allclose(solve(p, t, red), F.mse_loss(p, t, reduction=red), atol=1e-6), red


def check_invalid_reduction(solve):
    try:
        solve(torch.ones(3), torch.ones(3), "bogus")
    except ValueError:
        return
    raise AssertionError("非法 reduction 应抛出 ValueError")


def check_float64(solve):
    torch.manual_seed(42)
    p = torch.randn(16, dtype=torch.float64)
    t = torch.randn(16, dtype=torch.float64)
    out = solve(p, t, "none")
    assert out.dtype == torch.float64


check_mean.example = True
check_none.example = True
