import torch
import torch.nn.functional as F


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic(solve):
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    w = torch.ones(4)
    b = torch.zeros(4)
    y = solve(x, w, b)
    exp = F.layer_norm(x, (4,), w, b, 1e-5)
    assert torch.allclose(y, exp, atol=1e-6), f"got {y.tolist()}, expected {exp.tolist()}"


def check_matches_torch(solve):
    torch.manual_seed(42)
    x = torch.randn(8, 16)
    w = torch.randn(16)
    b = torch.randn(16)
    y = solve(x, w, b)
    assert torch.allclose(y, F.layer_norm(x, (16,), w, b, 1e-5), atol=1e-5), "与 F.layer_norm 不一致"


def check_multi_dim_norm(solve):
    # normalized_shape 为尾部两维 (C, H)：LayerNorm 可一次归一化多个尾维
    torch.manual_seed(0)
    x = torch.randn(2, 3, 4)
    w = torch.randn(3, 4)
    b = torch.randn(3, 4)
    y = solve(x, w, b)
    assert torch.allclose(y, F.layer_norm(x, (3, 4), w, b, 1e-5), atol=1e-5), "多维 normalized_shape 不一致"


def check_stats_per_row(solve):
    # 每行独立归一化：输出每行均值≈0、方差≈1（w=1,b=0 时）
    torch.manual_seed(7)
    x = torch.randn(5, 32) * 10 + 100
    y = solve(x, torch.ones(32), torch.zeros(32))
    assert torch.allclose(y.mean(dim=-1), torch.zeros(5), atol=1e-5), "行均值应为 0"
    assert torch.allclose(y.var(dim=-1, unbiased=False), torch.ones(5), rtol=1e-3), "行方差应≈1"


def check_eps_guard(solve):
    # 常数向量：var=0，无 eps 会除零得 NaN
    x = torch.full((1, 8), 3.0)
    y = solve(x, torch.ones(8), torch.zeros(8), eps=1e-5)
    assert torch.isfinite(y).all(), "eps 未加导致 NaN/inf"
    assert torch.allclose(y, torch.zeros(1, 8), atol=1e-6), "常数向量归一化后应≈0"


def check_float64(solve):
    x = torch.randn(4, 8, dtype=torch.float64)
    w = torch.randn(8, dtype=torch.float64)
    b = torch.randn(8, dtype=torch.float64)
    y = solve(x, w, b)
    assert y.dtype == torch.float64
    assert torch.allclose(y, F.layer_norm(x, (8,), w, b, 1e-5), atol=1e-10)


check_basic.example = True
