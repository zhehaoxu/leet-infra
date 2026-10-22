import torch


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic(solve):
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    y = solve(a, b)
    expected = torch.tensor([[19.0, 22.0], [43.0, 50.0]])
    assert torch.allclose(y, expected), f"got {y.tolist()}, expected {expected.tolist()}"


def check_bmm_matches(solve):
    torch.manual_seed(42)
    a = torch.randn(6, 4, 3)
    b = torch.randn(6, 3, 5)
    y = solve(a, b)
    assert torch.allclose(y, torch.bmm(a, b), atol=1e-5), "与 torch.bmm 不一致"


def check_matmul_broadcast(solve):
    torch.manual_seed(0)
    # 2D @ 3D、3D @ 2D、batch 维含 1 的广播，全部对齐 torch.matmul
    m = torch.randn(4, 3)
    batched = torch.randn(6, 3, 5)
    tail = torch.randn(5, 2)
    assert torch.allclose(solve(m, batched), torch.matmul(m, batched), atol=1e-5)
    assert torch.allclose(solve(batched, tail), torch.matmul(batched, tail), atol=1e-5)
    a = torch.randn(2, 1, 4, 3)
    b = torch.randn(2, 6, 3, 5)
    y = solve(a, b)
    assert y.shape == (2, 6, 4, 5), f"广播形状错误 {tuple(y.shape)}"
    assert torch.allclose(y, torch.matmul(a, b), atol=1e-5), "batch 广播不一致"


def check_noncontiguous_views(solve):
    # 转置视图非连续：元素 stride 与内存布局不一致时结果仍须正确
    torch.manual_seed(1)
    a = torch.randn(3, 8, 6).transpose(1, 2)  # (3,6,8) 视图
    b = torch.randn(3, 6, 8).mT               # (3,8,6) 视图
    y = solve(a, b)
    assert torch.allclose(y, torch.matmul(a, b), atol=1e-5), "非连续视图结果错误"


def check_4d_and_dtype(solve):
    torch.manual_seed(2)
    a = torch.randn(2, 3, 4, 5, dtype=torch.float64)
    b = torch.randn(2, 3, 5, 4, dtype=torch.float64)
    y = solve(a, b)
    assert y.dtype == torch.float64
    assert torch.allclose(y, torch.matmul(a, b), atol=1e-10), "4D matmul 不一致"


def check_shape_mismatch_raises(solve):
    a = torch.randn(2, 3, 4)
    b = torch.randn(2, 5, 6)
    try:
        solve(a, b)
    except ValueError:
        return
    raise AssertionError("内维不匹配应抛 ValueError")


check_basic.example = True
