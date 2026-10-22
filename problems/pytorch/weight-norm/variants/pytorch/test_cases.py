import torch
import torch.nn as nn
import torch.nn.utils as utils


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic(solve):
    g = torch.tensor([[1.0], [2.0]])
    v = torch.tensor([[3.0, 4.0], [3.0, 4.0]])
    y = solve(g, v, dim=1)
    expected = torch.tensor([[0.6, 0.8], [1.2, 1.6]])
    assert torch.allclose(y, expected, atol=1e-6), f"got {y.tolist()}, expected {expected.tolist()}"


def check_matches_torch(solve):
    # 与 torch.nn.utils.weight_norm 展开后的 weight 对拍（dim=0，Linear 权重）
    torch.manual_seed(42)
    v = torch.randn(4, 6)
    g = v.norm(p=2, dim=0, keepdim=True)  # weight_norm 初始化时 g = ||v||
    lin = nn.Linear(6, 4)
    lin.weight.data = v.clone()
    wn = utils.weight_norm(lin, dim=0)
    y = solve(g, v, dim=0)
    assert torch.allclose(y, wn.weight.data, atol=1e-6), "与 nn.utils.weight_norm 展开结果不一致"


def check_norm_equals_g(solve):
    # 核心不变量：沿 dim 的范数恰为 |g|（非零切片）
    torch.manual_seed(0)
    v = torch.randn(5, 7)
    g = torch.randn(5, 1)
    w = solve(g, v, dim=1)
    assert torch.allclose(w.norm(dim=1, keepdim=True), g.abs(), atol=1e-5), "||w||_dim 应等于 |g|"


def check_direction_preserved(solve):
    # 重参数化只改长度不改方向：w 与 v 的余弦相似度为 1
    torch.manual_seed(1)
    v = torch.randn(4, 8)
    g = torch.rand(4, 1) + 0.5
    w = solve(g, v, dim=1)
    cos = torch.nn.functional.cosine_similarity(w, v, dim=1)
    assert torch.allclose(cos, torch.ones(4), atol=1e-5), "方向应保持"


def check_zero_slice(solve):
    # 全零切片：clamp_min 防除零，输出应为 0 而非 NaN
    v = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
    g = torch.ones(2, 1)
    w = solve(g, v, dim=1)
    assert torch.isfinite(w).all(), "全零切片产生 NaN/inf"
    assert torch.allclose(w, torch.tensor([[0.6, 0.8], [0.0, 0.0]]), atol=1e-6)


def check_float64(solve):
    v = torch.randn(3, 5, dtype=torch.float64)
    g = torch.randn(3, 1, dtype=torch.float64)
    w = solve(g, v, dim=1)
    assert w.dtype == torch.float64
    assert torch.allclose(w.norm(dim=1, keepdim=True), g.abs(), atol=1e-12)


check_basic.example = True
