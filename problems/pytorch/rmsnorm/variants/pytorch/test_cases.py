import torch


def _manual_rmsnorm(x, weight, eps):
    rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x / rms * weight


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic(solve):
    x = torch.tensor([[1.0, 2.0, 3.0]])
    w = torch.ones(3)
    y = solve(x, w, eps=1e-6)
    exp = _manual_rmsnorm(x, w, 1e-6)
    assert torch.allclose(y, exp, atol=1e-6), f"got {y.tolist()}, expected {exp.tolist()}"


def check_matches_formula(solve):
    torch.manual_seed(42)
    x = torch.randn(8, 64)
    w = torch.randn(64)
    y = solve(x, w, eps=1e-6)
    assert torch.allclose(y, _manual_rmsnorm(x, w, 1e-6), atol=1e-5), "与手推公式不一致"


def check_no_mean_centering(solve):
    # RMSNorm 不减均值：全正输入的输出不应关于 0 对称（区别于 LayerNorm）
    x = torch.full((1, 4), 2.0)
    y = solve(x, torch.ones(4), eps=1e-6)
    assert (y > 0).all(), "输出应保持全正——RMSNorm 不做均值中心化"
    assert torch.allclose(y, torch.full((1, 4), 2.0 / torch.sqrt(torch.tensor(4.0 + 1e-6))), atol=1e-6)


def check_scale_invariance(solve):
    # 输入整体放大 10 倍，输出不变（缩放不变性是 RMSNorm 的核心性质）
    torch.manual_seed(0)
    x = torch.randn(4, 32)
    w = torch.randn(32)
    y1 = solve(x, w)
    y2 = solve(x * 10.0, w)
    assert torch.allclose(y1, y2, atol=1e-4), "RMSNorm 应对输入尺度不变"


def check_zero_vector(solve):
    # 全零行：eps 保护，不得出现 NaN
    x = torch.zeros(2, 8)
    x[1] = torch.randn(8)
    y = solve(x, torch.ones(8), eps=1e-6)
    assert torch.isfinite(y).all(), "全零输入未加 eps 导致 NaN"
    assert torch.allclose(y[0], torch.zeros(8), atol=1e-6), "全零行输出应为 0"


def check_float64(solve):
    x = torch.randn(4, 16, dtype=torch.float64)
    w = torch.randn(16, dtype=torch.float64)
    y = solve(x, w)
    assert y.dtype == torch.float64
    assert torch.allclose(y, _manual_rmsnorm(x, w, 1e-6), atol=1e-10)


check_basic.example = True
