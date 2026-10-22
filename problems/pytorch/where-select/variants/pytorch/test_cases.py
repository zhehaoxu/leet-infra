import torch


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic(solve):
    condition = torch.tensor([True, False, True, False])
    x = torch.tensor([1, 2, 3, 4])
    y = torch.tensor([-1, -2, -3, -4])
    y_out = solve(condition, x, y)
    assert torch.equal(y_out, torch.tensor([1, -2, 3, -4])), f"got {y_out.tolist()}"


def check_matches_torch(solve):
    torch.manual_seed(0)
    a = torch.randn(3, 4)
    b = torch.randn(3, 4)
    cond = torch.randn(3, 4) > 0
    assert torch.equal(solve(cond, a, b), torch.where(cond, a, b))


def check_scalar_branches(solve):
    x = torch.tensor([1.0, -2.0, 3.0, -4.0])
    cond = x > 0
    # 负值替换为 0（标量 y）
    y = solve(cond, x, torch.zeros(()))
    assert torch.equal(y, torch.where(cond, x, 0.0)), "标量分支应可广播"


def check_broadcast_shapes(solve):
    cond = torch.tensor([[True], [False], [True]])       # (3,1)
    x = torch.tensor([[1, 2, 3]])                          # (1,3)
    y = torch.zeros(3, 3)
    out = solve(cond, x, y)
    assert out.shape == (3, 3)
    assert torch.equal(out, torch.where(cond, x, y)), "condition/x/y 需按广播对齐"


def check_dtype_promotion(solve):
    cond = torch.tensor([True, False])
    out = solve(cond, torch.tensor([1, 2]), torch.tensor([0.5, 0.5]))
    assert out.dtype == torch.float32 and torch.allclose(out, torch.tensor([1.0, 0.5]))


check_basic.example = True
