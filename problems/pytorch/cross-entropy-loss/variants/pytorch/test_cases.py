import torch
import torch.nn.functional as F


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_mean(solve):
    logits = torch.tensor([[2.0, 1.0, 0.1], [0.1, 2.0, 1.0], [1.0, 0.5, 3.0]])
    target = torch.tensor([0, 1, 2])
    out = solve(logits, target)
    assert abs(out.item() - 0.343598) < 1e-4, f"got {out.item()}"


def check_none(solve):
    logits = torch.tensor([[2.0, 1.0, 0.1], [0.1, 2.0, 1.0], [1.0, 0.5, 3.0]])
    target = torch.tensor([0, 1, 2])
    out = solve(logits, target, "none")
    expected = torch.tensor([0.41703, 0.41703, 0.19673])
    assert torch.allclose(out, expected, atol=1e-4), f"got {out.tolist()}"


def check_stable_big_logits(solve):
    logits = torch.tensor([[1000.0, 999.0, 1001.0]])
    target = torch.tensor([1])
    out = solve(logits, target)
    assert torch.isfinite(out), "大 logits 下必须有限（稳定式）"
    assert abs(out.item() - 2.407605) < 1e-4, f"got {out.item()}"


def check_matches_F(solve):
    torch.manual_seed(42)
    logits = torch.randn(256, 10)
    target = torch.randint(0, 10, (256,))
    for red in ("mean", "sum", "none"):
        assert torch.allclose(solve(logits, target, red), F.cross_entropy(logits, target, reduction=red), atol=1e-5), red


def check_invalid_reduction(solve):
    try:
        solve(torch.randn(2, 3), torch.tensor([0, 1]), "bogus")
    except ValueError:
        return
    raise AssertionError("非法 reduction 应抛出 ValueError")


check_mean.example = True
check_none.example = True
check_stable_big_logits.example = True
