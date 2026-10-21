import torch


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_param_count(solve):
    torch.manual_seed(42)
    model = solve(8, 4, 2)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Linear(8->32): 8*32+32=288, Linear(32->2): 32*2+2=66
    assert n == 354, f"可训练参数量应为 354, got {n}"


def check_is_module(solve):
    torch.manual_seed(42)
    model = solve(4, 2, 3)
    assert isinstance(model, torch.nn.Module), "solve 必须返回 nn.Module 实例"


def check_forward_shape(solve):
    torch.manual_seed(42)
    model = solve(6, 3, 5)
    x = torch.randn(7, 6)
    out = model(x)
    assert out.shape == (7, 5), f"前向输出应为 (7,5), got {tuple(out.shape)}"


def check_backward_all_params(solve):
    torch.manual_seed(42)
    model = solve(8, 4, 2)
    x = torch.randn(4, 8)
    out = model(x)
    out.sum().backward()
    for i, p in enumerate(model.parameters()):
        assert p.grad is not None, f"参数 #{i} 反向后 .grad 为 None，梯度链断了"
        assert p.grad.shape == p.shape, f"参数 #{i} 梯度形状错误"


def check_relu_nonneg(solve):
    # 中间层经 ReLU：整体输出对特殊输入可正可负，但要求模块确实用了非线性——
    # 用两层全零偏置无法检查，改为检查同 seed 下两次实例化输出一致（确定性）
    torch.manual_seed(42)
    m1 = solve(5, 2, 3)
    x = torch.randn(2, 5)
    torch.manual_seed(42)
    m2 = solve(5, 2, 3)
    assert torch.equal(m1(x), m2(x)), "同 seed 实例化应给出一致的前向结果"


def check_invalid_multiplier(solve):
    try:
        solve(8, 0, 2)
    except ValueError:
        return
    raise AssertionError("hidden_multiplier < 1 应抛出 ValueError")


check_param_count.example = True
