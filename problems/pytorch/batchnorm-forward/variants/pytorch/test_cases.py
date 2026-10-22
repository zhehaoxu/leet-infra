import torch
import torch.nn.functional as F


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic(solve):
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # (N=2, C=2)
    w = torch.ones(2)
    b = torch.zeros(2)
    rm = torch.zeros(2)
    rv = torch.ones(2)
    y, nrm, nrv = solve(x, w, b, rm, rv, training=True, momentum=0.1)
    exp_rm = rm.clone()
    exp_rv = rv.clone()
    exp_y = F.batch_norm(x, exp_rm, exp_rv, w, b, True, 0.1, 1e-5)
    assert torch.allclose(y, exp_y, atol=1e-6), f"got {y.tolist()}, expected {exp_y.tolist()}"
    assert torch.allclose(nrm, exp_rm, atol=1e-6), f"running_mean got {nrm.tolist()}, expected {exp_rm.tolist()}"
    assert torch.allclose(nrv, exp_rv, atol=1e-6), f"running_var got {nrv.tolist()}, expected {exp_rv.tolist()}"


def check_matches_torch(solve):
    torch.manual_seed(42)
    x = torch.randn(8, 4, 6, 6)
    w = torch.randn(4)
    b = torch.randn(4)
    rm = torch.randn(4)
    rv = torch.rand(4) + 0.5
    ref_rm = rm.clone()
    ref_rv = rv.clone()
    exp_y = F.batch_norm(x, ref_rm, ref_rv, w, b, True, 0.1, 1e-5)
    y, nrm, nrv = solve(x, w, b, rm, rv, training=True, momentum=0.1)
    assert torch.allclose(y, exp_y, atol=1e-5), "训练模式与 F.batch_norm 不一致"
    assert torch.allclose(nrm, ref_rm, atol=1e-5), "running_mean 更新与 F.batch_norm 不一致"
    assert torch.allclose(nrv, ref_rv, atol=1e-5), "running_var 应按无偏方差更新"


def check_eval_mode(solve):
    torch.manual_seed(1)
    x = torch.randn(4, 3, 5, 5)
    w = torch.randn(3)
    b = torch.randn(3)
    rm = torch.randn(3)
    rv = torch.rand(3) + 0.5
    exp_y = F.batch_norm(x, rm.clone(), rv.clone(), w, b, False, 0.1, 1e-5)
    y, nrm, nrv = solve(x, w, b, rm, rv, training=False)
    assert torch.allclose(y, exp_y, atol=1e-5), "推理模式应使用 running stats"
    assert torch.equal(nrm, rm) and torch.equal(nrv, rv), "推理模式不应修改 running stats"


def check_momentum_update(solve):
    torch.manual_seed(2)
    x = torch.randn(16, 2)
    w = torch.ones(2)
    b = torch.zeros(2)
    rm = torch.full((2,), 10.0)
    rv = torch.full((2,), 10.0)
    _, nrm, nrv = solve(x, w, b, rm, rv, training=True, momentum=0.3)
    mu = x.mean(0)
    assert torch.allclose(nrm, 0.7 * rm + 0.3 * mu, atol=1e-6), "momentum 更新公式错误"
    assert torch.allclose(nrv, 0.7 * rv + 0.3 * x.var(0, unbiased=True), atol=1e-6), "momentum 更新公式错误"


def check_var_running_unbiased(solve):
    # 陷阱：归一化用有偏方差，但 running_var 存的是无偏方差（除 N-1）
    torch.manual_seed(3)
    x = torch.randn(8, 3)
    z = torch.zeros(3)
    ones = torch.ones(3)
    _, _, nrv = solve(x, ones, z, z.clone(), ones.clone(), training=True, momentum=0.5)
    expected = 0.5 * ones + 0.5 * x.var(0, unbiased=True)
    assert torch.allclose(nrv, expected, atol=1e-6), "running_var 应使用无偏方差(unbiased=True)"


check_basic.example = True
