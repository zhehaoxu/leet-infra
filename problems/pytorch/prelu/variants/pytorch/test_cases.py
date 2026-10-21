import torch


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic(solve):
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    w = torch.tensor(0.25)  # 标量权重
    out = solve(x, w)
    expected = torch.tensor([-0.5, -0.25, 0.0, 1.0, 2.0])
    assert torch.allclose(out, expected, atol=1e-6), f"got {out.tolist()}, expected {expected.tolist()}"


def check_per_channel_weight(solve):
    # weight 形状 (C,)，x 形状 (N, C)：按通道广播
    x = torch.tensor([[-2.0, -4.0, 3.0], [-1.0, -5.0, 6.0]])
    w = torch.tensor([0.1, 0.5, 0.9])
    out = solve(x, w)
    expected = torch.tensor([[-0.2, -2.0, 3.0], [-0.1, -2.5, 6.0]])
    assert torch.allclose(out, expected, atol=1e-6), f"逐通道错误: got {out.tolist()}"


def check_matches_torch(solve):
    torch.manual_seed(42)
    x = torch.randn(8, 4, 6, 6)
    w = torch.rand(4) * 0.5  # 通道维=1，长度 C=4
    assert torch.allclose(solve(x, w), torch.nn.functional.prelu(x, w), atol=1e-6), "与 F.prelu 不一致"


def check_zero_weight_is_relu(solve):
    torch.manual_seed(0)
    x = torch.randn(32)
    w = torch.tensor(0.0)
    assert torch.allclose(solve(x, w), torch.relu(x)), "weight=0 应退化为 ReLU"


def check_shape_preserved(solve):
    x = torch.randn(5, 3, 2)
    w = torch.rand(3)
    out = solve(x, w)
    assert out.shape == x.shape and out.dtype == x.dtype, "形状/dtype 必须保持"


check_basic.example = True
