import torch


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic(solve):
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    out = solve(x, dim=-1)
    # a=[1,2], b=[3,4]; sigmoid(3)=0.95257, sigmoid(4)=0.98201
    expected = torch.tensor([0.95257413, 1.96402752])
    assert out.shape == (2,), f"末维应减半为 2, got {tuple(out.shape)}"
    assert torch.allclose(out, expected, atol=1e-6), f"got {out.tolist()}, expected {expected.tolist()}"


def check_matches_torch(solve):
    import torch.nn.functional as F

    torch.manual_seed(42)
    x = torch.randn(4, 8)
    assert torch.allclose(solve(x, dim=-1), F.glu(x, dim=-1), atol=1e-6), "dim=-1 与 F.glu 不一致"
    assert torch.allclose(solve(x, dim=0), F.glu(x, dim=0), atol=1e-6), "dim=0 与 F.glu 不一致"


def check_output_shape(solve):
    x = torch.randn(2, 6, 10)
    out = solve(x, dim=-1)
    assert out.shape == (2, 6, 5), f"expected (2,6,5), got {tuple(out.shape)}"


def check_odd_dim_raises(solve):
    # 末维为奇数无法对半切，chunk 会抛异常（RuntimeError/ValueError）
    x = torch.randn(3, 5)
    try:
        solve(x, dim=-1)
    except Exception:
        return
    raise AssertionError("奇数维应无法对半切分并报错")


check_basic.example = True
