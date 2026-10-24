import torch


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_first_step(solve):
    p = torch.tensor([1.0, 1.0, 1.0])
    g = torch.tensor([0.5, -0.5, 2.0])
    b = torch.zeros(3)
    np_, nb = solve(p, g, b, lr=0.1, momentum=0.9)
    assert torch.allclose(np_, torch.tensor([0.95, 1.05, 0.80]), atol=1e-5), f"got {np_.tolist()}"
    assert torch.allclose(nb, torch.tensor([0.5, -0.5, 2.0]), atol=1e-6), f"buf {nb.tolist()}"


def check_no_momentum(solve):
    p = torch.tensor([2.0, -1.0])
    g = torch.tensor([1.0, 1.0])
    b = torch.zeros(2)
    np_, nb = solve(p, g, b, lr=0.5, momentum=0.0)
    assert torch.allclose(np_, torch.tensor([1.5, -1.5]), atol=1e-6), f"got {np_.tolist()}"
    assert torch.allclose(nb, torch.zeros(2)), "momentum=0 时 buffer 应保持不变（零）"


def check_weight_decay(solve):
    # 无动量、wd=0.1：g' = g + 0.1*p = [0.6, -0.4, 2.1]; p' = p - 0.1*g'
    p = torch.tensor([1.0, 1.0, 1.0])
    g = torch.tensor([0.5, -0.5, 2.0])
    b = torch.zeros(3)
    np_, _ = solve(p, g, b, lr=0.1, momentum=0.0, weight_decay=0.1)
    expected = p - 0.1 * (g + 0.1 * p)
    assert torch.allclose(np_, expected, atol=1e-6), f"got {np_.tolist()}"


def check_buffer_carryover(solve):
    # 连续两步，验证 buffer 累积
    p = torch.tensor([1.0, 1.0, 1.0])
    b = torch.zeros(3)
    p, b = solve(p, torch.tensor([0.5, -0.5, 2.0]), b, lr=0.1, momentum=0.9)
    p, b = solve(p, torch.tensor([0.5, 0.5, 0.5]), b, lr=0.1, momentum=0.9)
    assert torch.allclose(b, torch.tensor([0.95, 0.05, 2.3]), atol=1e-5), f"buf {b.tolist()}"
    assert torch.allclose(p, torch.tensor([0.855, 1.045, 0.57]), atol=1e-5), f"p {p.tolist()}"


def check_matches_torch_optim(solve):
    torch.manual_seed(42)
    p_ref = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.SGD([p_ref], lr=0.1, momentum=0.9, weight_decay=0.02)
    p = p_ref.detach().clone()
    b = torch.zeros_like(p)
    for _ in range(10):
        g = torch.randn(3, dtype=torch.float64)
        p_ref.grad = g.clone()
        opt.step()
        p, b = solve(p, g, b, lr=0.1, momentum=0.9, weight_decay=0.02)
    assert torch.allclose(p, p_ref.detach(), atol=1e-9), f"偏离 torch.optim.SGD: {p} vs {p_ref.detach()}"


def check_no_inplace(solve):
    p = torch.tensor([1.0, 1.0])
    g = torch.tensor([0.5, 0.5])
    b = torch.zeros(2)
    solve(p, g, b, lr=0.1, momentum=0.9)
    assert torch.equal(p, torch.tensor([1.0, 1.0])), "不应原地修改 params"


check_first_step.example = True
check_no_momentum.example = True
