import torch


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_first_step(solve):
    p = torch.tensor([1.0, 1.0])
    g = torch.tensor([0.1, -0.2])
    m = torch.zeros(2)
    v = torch.zeros(2)
    np_, nm, nv = solve(p, g, m, v, step=1, lr=0.1)
    assert torch.allclose(np_, torch.tensor([0.9, 1.1]), atol=1e-5), f"got {np_.tolist()}"
    assert torch.allclose(nm, torch.tensor([0.01, -0.02]), atol=1e-6), f"m {nm.tolist()}"
    assert torch.allclose(nv, torch.tensor([1e-5, 4e-5]), atol=1e-8), f"v {nv.tolist()}"


def check_sign_first_step(solve):
    # 首步等价于 p - lr*sign(g)（偏差修正的性质）
    p = torch.tensor([5.0, -3.0, 2.0])
    g = torch.tensor([0.7, -0.1, 0.0003])
    m = torch.zeros(3)
    v = torch.zeros(3)
    np_, _, _ = solve(p, g, m, v, step=1, lr=0.1)
    assert torch.allclose(np_, p - 0.1 * torch.sign(g), atol=1e-4), f"got {np_.tolist()}"


def check_state_carryover(solve):
    p = torch.tensor([1.0])
    m = torch.zeros(1)
    v = torch.zeros(1)
    p, m, v = solve(p, torch.tensor([0.5]), m, v, step=1, lr=0.1)
    p2, m2, v2 = solve(p, torch.tensor([-0.3]), m, v, step=2, lr=0.1)
    # m2 = 0.9*m1 + 0.1*(-0.3); v2 = 0.999*v1 + 0.001*0.09
    assert abs(m2.item() - (0.9 * 0.05 + 0.1 * (-0.3))) < 1e-6, f"m2 {m2.item()}"
    assert abs(v2.item() - (0.999 * 0.001 * 0.25 + 0.001 * 0.09)) < 1e-9, f"v2 {v2.item()}"


def check_matches_torch_optim(solve):
    torch.manual_seed(42)
    p_ref = torch.tensor([0.3, -1.2, 2.0], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([p_ref], lr=1e-2, betas=(0.9, 0.999), eps=1e-8)
    p = p_ref.detach().clone()
    m = torch.zeros_like(p)
    v = torch.zeros_like(p)
    for t in range(1, 21):
        g = torch.randn(3, dtype=torch.float64)
        p_ref.grad = g.clone()
        opt.step()
        p, m, v = solve(p, g, m, v, step=t, lr=1e-2)
    assert torch.allclose(p, p_ref.detach(), atol=1e-7), f"偏离 torch.optim.Adam: {p} vs {p_ref.detach()}"


def check_no_inplace(solve):
    p = torch.tensor([1.0, 2.0])
    g = torch.tensor([0.1, 0.2])
    m = torch.zeros(2)
    v = torch.zeros(2)
    solve(p, g, m, v, step=1, lr=0.1)
    assert torch.equal(p, torch.tensor([1.0, 2.0])), "不应原地修改 params"
    assert torch.equal(m, torch.zeros(2)), "不应原地修改 exp_avg"


check_first_step.example = True
check_sign_first_step.example = True
