import torch


def _step_all(opt, grads_seq):
    for gs in grads_seq:
        for p, g in zip(opt.param_groups[0]["params"], gs):
            p.grad = g
        opt.step()


def _ns(G, steps):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    tr = X.size(-2) > X.size(-1)
    if tr:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        X = a * X + (b * A + c * (A @ A)) @ X
    return X.mT if tr else X


def _oracle_muon(params, grads_seq, lr, mu, ns_steps, wd):
    """独立参照：向量化重算（与 Optimizer 类不同的实现路径）。"""
    ps = [p.clone() for p in params]
    bufs = [torch.zeros_like(p) for p in ps]
    for gs in grads_seq:
        for i, (p, gg) in enumerate(zip(ps, gs)):
            bufs[i] = mu * bufs[i] + (1 - mu) * gg
            G = (1 - mu) * gg + mu * bufs[i]
            O = _ns(G, ns_steps)
            O = O * max(1, O.size(-2) / O.size(-1)) ** 0.5
            if wd != 0:
                p.mul_(1 - lr * wd)
            p.sub_(lr * O.reshape(p.shape))
    return ps


def check_example(solve):
    ps = [torch.tensor([[0.5, -0.2, 0.1], [0.3, 0.4, -0.1]])]
    grads = [
        [torch.tensor([[0.2, -0.1, 0.05], [0.1, 0.3, -0.2]])],
        [torch.tensor([[-0.15, 0.2, 0.1], [0.05, -0.1, 0.3]])],
        [torch.tensor([[0.1, 0.05, -0.2], [-0.3, 0.15, 0.1]])],
    ]
    opt = solve(ps, lr=0.1, momentum=0.95, ns_steps=5, weight_decay=0.0)
    _step_all(opt, grads)
    exp = torch.tensor([[0.43220454454421997, -0.3229876458644867, 0.1247878223657608],
                        [0.256377249956131, 0.28521767258644104, -0.13744311034679413]])
    assert torch.allclose(ps[0], exp, atol=1e-5, rtol=1e-5), f"got {ps[0].tolist()}"


check_example.example = True


def check_matches_oracle_tall(solve):
    # rows>cols：触发转置分支，且谱缩放 = sqrt(rows/cols)
    torch.manual_seed(42)
    p0 = [torch.randn(5, 2)]
    gseq = [[torch.randn(5, 2) * 0.4] for _ in range(12)]
    ps = [p.clone() for p in p0]
    opt = solve(ps, lr=0.05, momentum=0.95, ns_steps=5, weight_decay=0.1)
    _step_all(opt, gseq)
    exp = _oracle_muon([p.clone() for p in p0], gseq, 0.05, 0.95, 5, 0.1)
    assert torch.allclose(ps[0], exp[0], atol=1e-5, rtol=1e-5), f"偏离参照:\n{ps[0]}\nvs\n{exp[0]}"


def check_matches_oracle_multi_square(solve):
    # 多个方阵（含两个不同尺寸），scale=1，多参数同组
    torch.manual_seed(1)
    p0 = [torch.randn(4, 4), torch.randn(3, 3)]
    gseq = [[torch.randn_like(p) for p in p0] for _ in range(15)]
    ps = [p.clone() for p in p0]
    opt = solve(ps, lr=0.02, momentum=0.9, ns_steps=5, weight_decay=0.0)
    _step_all(opt, gseq)
    exp = _oracle_muon([p.clone() for p in p0], gseq, 0.02, 0.9, 5, 0.0)
    for a, b in zip(ps, exp):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-5)


def check_none_grad_skipped(solve):
    ps = [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]])]
    opt = solve(ps, lr=0.1, momentum=0.95, weight_decay=0.1)
    opt.param_groups[0]["params"][0].grad = torch.tensor([[0.1, -0.1], [0.2, 0.3]])
    opt.step()
    assert torch.equal(ps[1], torch.tensor([[5.0, 6.0], [7.0, 8.0]])), "无梯度参数不应被改动"


def check_returns_optimizer_instance(solve):
    opt = solve([torch.randn(3, 3)], lr=0.02)
    assert isinstance(opt, torch.optim.Optimizer)
