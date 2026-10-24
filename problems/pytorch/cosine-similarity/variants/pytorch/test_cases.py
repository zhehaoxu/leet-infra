import torch
import torch.nn.functional as F


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic(solve):
    a = torch.tensor([[1.0, 2, 3]])
    b = torch.tensor([[4.0, 5, 6]])
    y = solve(a, b)
    assert abs(y.item() - 0.9746318) < 1e-5, y.item()


def check_matches_F(solve):
    torch.manual_seed(0)
    a = torch.randn(5, 8)
    b = torch.randn(5, 8)
    assert torch.allclose(solve(a, b), F.cosine_similarity(a, b, dim=-1), atol=1e-6)


def check_scale_invariance(solve):
    # 正数缩放不改变方向：cos(a, 7a) == 1
    torch.manual_seed(1)
    a = torch.randn(4, 6)
    assert torch.allclose(solve(a, a * 7.0), torch.ones(4), atol=1e-5)


def check_orthogonal_zero(solve):
    a = torch.tensor([[1.0, 0]])
    b = torch.tensor([[0.0, 1]])
    assert abs(solve(a, b).item()) < 1e-7


def check_zero_vector_safe(solve):
    # 零向量不应产生 nan/inf（被 eps 兜住）
    a = torch.zeros(1, 4)
    b = torch.randn(1, 4)
    y = solve(a, b)
    assert torch.isfinite(y).all(), "零向量应被 eps 保护，不出现 nan/inf"


check_basic.example = True
