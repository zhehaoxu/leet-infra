import torch


def _step_all(opt, grads_seq):
    for gs in grads_seq:
        for p, g in zip(opt.param_groups[0]["params"], gs):
            p.grad = g
        opt.step()


def _oracle_lion(params, grads_seq, lr, betas, wd):
    """独立参照：向量化重算（与 Optimizer 类不同的实现路径）。"""
    b1, b2 = betas
    ps = [p.clone() for p in params]
    ms = [torch.zeros_like(p) for p in ps]
    for gs in grads_seq:
        for i, (p, gg) in enumerate(zip(ps, gs)):
            if wd != 0:
                p.mul_(1 - lr * wd)
            u = (b1 * ms[i] + (1 - b1) * gg).sign()
            ms[i] = b2 * ms[i] + (1 - b2) * gg
            p.sub_(lr * u)
    return ps


def check_example(solve):
    ps = [torch.tensor([1.0, -2.0, 0.5])]
    grads = [
        [torch.tensor([0.1, -0.2, 0.3])],
        [torch.tensor([-0.4, 0.05, 0.2])],
        [torch.tensor([0.15, -0.1, -0.25])],
    ]
    opt = solve(ps, lr=0.1, betas=(0.9, 0.99), weight_decay=0.01)
    _step_all(opt, grads)
    exp = torch.tensor([0.8971028923988342, -1.8941060304641724, 0.39880141615867615])
    assert torch.allclose(ps[0], exp, atol=1e-5, rtol=1e-5), f"got {ps[0].tolist()}"


check_example.example = True


def check_matches_oracle(solve):
    torch.manual_seed(42)
    p0 = [torch.randn(4, 5), torch.randn(3)]
    gseq = [[torch.randn_like(p) for p in p0] for _ in range(25)]
    ps = [p.clone() for p in p0]
    opt = solve(ps, lr=5e-3, betas=(0.9, 0.99), weight_decay=0.02)
    _step_all(opt, gseq)
    exp = _oracle_lion([p.clone() for p in p0], gseq, 5e-3, (0.9, 0.99), 0.02)
    for a, b in zip(ps, exp):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-5), f"偏离参照:\n{a}\nvs\n{b}"


def check_unit_step_sign(solve):
    # 单步、零初始动量时 u=sign((1−β1)g)=sign(g)，位移恰为 lr·sign(g)（wd=0）
    ps = [torch.tensor([2.0, -3.0, 0.0])]
    _step_all(solve(ps, lr=0.1, betas=(0.9, 0.99), weight_decay=0.0), [[torch.tensor([0.5, -0.1, 0.0])]])
    # 第三个坐标 g=0 → u=sign(0)=0 → 不动
    exp = torch.tensor([2.0 - 0.1, -3.0 + 0.1, 0.0])
    assert torch.allclose(ps[0], exp, atol=1e-6), f"got {ps[0].tolist()}"


def check_none_grad_skipped(solve):
    ps = [torch.tensor([1.0, 2.0]), torch.tensor([5.0, 6.0])]
    opt = solve(ps, lr=0.1, betas=(0.9, 0.99), weight_decay=0.1)
    opt.param_groups[0]["params"][0].grad = torch.tensor([0.1, -0.1])
    opt.step()
    assert torch.equal(ps[1], torch.tensor([5.0, 6.0])), "无梯度参数不应被改动"


def check_returns_optimizer_instance(solve):
    opt = solve([torch.tensor([1.0])], lr=0.1)
    assert isinstance(opt, torch.optim.Optimizer)
