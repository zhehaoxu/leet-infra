import torch


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_quadratic(solve):
    x = torch.tensor([1.0, 2.0])
    out = solve(x, "quadratic")
    expected = torch.tensor([8.0, 14.0])  # 6x+2
    assert out.shape == x.shape, f"expected {tuple(x.shape)}, got {tuple(out.shape)}"
    assert torch.allclose(out, expected), f"quadratic 梯度错误: got {out.tolist()}"


def check_cubic(solve):
    x = torch.tensor([-1.0, 0.5, 2.0])
    out = solve(x, "cubic")
    expected = 3 * x**2  # 3x^2
    assert torch.allclose(out, expected, atol=1e-6), f"cubic 梯度错误: got {out.tolist()}"


def check_exp(solve):
    x = torch.tensor([0.0, 1.0, -0.5])
    out = solve(x, "exp")
    expected = torch.exp(x)  # e^x
    assert torch.allclose(out, expected, atol=1e-6), f"exp 梯度错误: got {out.tolist()}"


def check_no_input_pollution(solve):
    # 不得对入参原地开启梯度追踪
    x = torch.tensor([1.0, 2.0, 3.0])
    solve(x, "quadratic")
    assert not x.requires_grad, "solve 不应把入参 x 设为 requires_grad"
    assert x.grad is None, "solve 不应在入参 x 上累积 .grad"


def check_2d_shape(solve):
    x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    out = solve(x, "quadratic")
    assert out.shape == (2, 3), f"expected (2,3), got {tuple(out.shape)}"
    assert torch.allclose(out, 6 * x + 2, atol=1e-5), f"2D 梯度错误: got {out.tolist()}"


def check_invalid_expr(solve):
    try:
        solve(torch.ones(2), "log")
    except ValueError:
        return
    raise AssertionError("非法 expr 应抛出 ValueError")


check_quadratic.example = True
