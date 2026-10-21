import torch


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic(solve):
    x = torch.tensor([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]])
    out = solve(x, dim=-1)
    expected = torch.tensor(
        [[0.09003, 0.24473, 0.66524], [0.33333, 0.33333, 0.33333]]
    )
    assert torch.allclose(out, expected, atol=1e-4), f"结果错误: got {out.tolist()}"


def check_simplex(solve):
    torch.manual_seed(42)
    x = torch.randn(16, 8)
    out = solve(x, dim=-1)
    assert (out >= 0).all(), "概率不应为负"
    sums = out.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6), "每行概率和应为 1"


def check_matches_torch_both_dims(solve):
    import torch.nn.functional as F

    torch.manual_seed(42)
    x = torch.randn(6, 10, 4)
    assert torch.allclose(solve(x, dim=-1), F.softmax(x, dim=-1), atol=1e-6), "dim=-1 与 torch 不一致"
    assert torch.allclose(solve(x, dim=0), F.softmax(x, dim=0), atol=1e-6), "dim=0 与 torch 不一致"
    assert torch.allclose(solve(x, dim=1), F.softmax(x, dim=1), atol=1e-6), "dim=1 与 torch 不一致"


def check_no_overflow_1000_scale(solve):
    # 朴素实现 exp(1000) = inf → inf/inf = NaN；减 max 后恒安全
    x = torch.tensor([[1000.0, 999.0, 1000.5], [-1000.0, 0.0, 1000.0]])
    out = solve(x, dim=-1)
    assert torch.isfinite(out).all(), f"出现 inf/NaN: {out.tolist()}"
    import torch.nn.functional as F

    assert torch.allclose(out, F.softmax(x, dim=-1), atol=1e-6), "千量级 logits 结果错误"


def check_argmax_preserved(solve):
    torch.manual_seed(42)
    x = torch.randn(32, 12)
    out = solve(x, dim=-1)
    assert torch.equal(out.argmax(dim=-1), x.argmax(dim=-1)), "softmax 应保持最大 logits 的位置"


check_basic.example = True
