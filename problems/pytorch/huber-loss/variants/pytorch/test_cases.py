import torch
import torch.nn.functional as F


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_mean(solve):
    pred = torch.tensor([1.5, -1.0, 0.3, 4.0])
    target = torch.tensor([1.0, 0.5, 0.2, 1.0])
    out = solve(pred, target, 1.0)
    assert abs(out.item() - 0.9075) < 1e-4, f"got {out.item()}"


def check_none(solve):
    pred = torch.tensor([1.5, -1.0, 0.3, 4.0])
    target = torch.tensor([1.0, 0.5, 0.2, 1.0])
    out = solve(pred, target, 1.0, "none")
    expected = torch.tensor([0.125, 1.0, 0.005, 2.5])
    assert torch.allclose(out, expected, atol=1e-4), f"got {out.tolist()}"


def check_delta_switch(solve):
    # 残差 2.0：delta=1 -> 线性支 1*(2-0.5)=1.5；delta=3 -> 二次支 0.5*4=2.0
    pred = torch.tensor([2.0])
    target = torch.tensor([0.0])
    assert abs(solve(pred, target, 1.0).item() - 1.5) < 1e-6
    assert abs(solve(pred, target, 3.0).item() - 2.0) < 1e-6


def check_matches_F(solve):
    torch.manual_seed(42)
    p = torch.randn(128, 16)
    t = torch.randn(128, 16)
    for d in (1.0, 0.5, 2.0):
        for red in ("mean", "sum", "none"):
            assert torch.allclose(solve(p, t, d, red), F.huber_loss(p, t, delta=d, reduction=red), atol=1e-6), (d, red)


def check_invalid_reduction(solve):
    try:
        solve(torch.ones(3), torch.zeros(3), 1.0, "bogus")
    except ValueError:
        return
    raise AssertionError("非法 reduction 应抛出 ValueError")


check_mean.example = True
check_none.example = True
check_delta_switch.example = True
