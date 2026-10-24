import torch
import torch.nn.functional as F


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_mean(solve):
    pred = torch.tensor([0.9, 0.2, 0.75, 0.4])
    target = torch.tensor([1.0, 0.0, 1.0, 0.0])
    out = solve(pred, target)
    assert abs(out.item() - 0.281753) < 1e-4, f"got {out.item()}"


def check_none(solve):
    pred = torch.tensor([0.9, 0.2, 0.75, 0.4])
    target = torch.tensor([1.0, 0.0, 1.0, 0.0])
    out = solve(pred, target, "none")
    expected = torch.tensor([0.10536, 0.22314, 0.28768, 0.51083])
    assert torch.allclose(out, expected, atol=1e-4), f"got {out.tolist()}"


def check_matches_F(solve):
    torch.manual_seed(42)
    p = torch.rand(64, 8).clamp(1e-3, 1 - 1e-3)
    t = (torch.rand(64, 8) > 0.5).float()
    for red in ("mean", "sum", "none"):
        assert torch.allclose(solve(p, t, red), F.binary_cross_entropy(p, t, reduction=red), atol=1e-5), red


def check_invalid_reduction(solve):
    try:
        solve(torch.full((3,), 0.5), torch.ones(3), "bogus")
    except ValueError:
        return
    raise AssertionError("非法 reduction 应抛出 ValueError")


def check_float64(solve):
    torch.manual_seed(42)
    p = torch.rand(16, dtype=torch.float64).clamp(1e-3, 1 - 1e-3)
    t = (torch.rand(16, dtype=torch.float64) > 0.5).float().double()
    out = solve(p, t, "none")
    assert out.dtype == torch.float64


check_mean.example = True
check_none.example = True
