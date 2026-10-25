import torch


def _step_all(opt, grads_seq):
    for gs in grads_seq:
        for p, g in zip(opt.param_groups[0]["params"], gs):
            p.grad = g
        opt.step()


def _approx_sq_grad(vr, vc):
    r = vr / vr.sum(dim=-1, keepdim=True)
    return r.unsqueeze(-1) * vc


def _oracle_adafactor(params, grads_seq, lr, betas, eps1, eps2, decay_rate, clip, scale_param, wd):
    """独立参照：向量化重算（与 Optimizer 类不同的实现路径）。"""
    b1, b2cap = betas
    ps = [p.clone() for p in params]
    st = [{} for _ in ps]
    for t, gs in enumerate(grads_seq, start=1):
        b2t = min(b2cap, 1.0 - t ** decay_rate)
        b = 1.0 - b2t
        for i, (p, gg) in enumerate(zip(ps, gs)):
            s = st[i]
            factored = gg.ndim >= 2
            if "m" not in s:
                s["m"] = torch.zeros_like(p)
                if factored:
                    s["vr"] = torch.zeros(gg.shape[:-1])
                    s["vc"] = torch.zeros(gg.shape[-1:])
                else:
                    s["v"] = torch.zeros_like(p)
            lr_eff = lr
            if scale_param:
                lr_eff = max(eps2, p.pow(2).mean().sqrt().item()) * lr
            U = gg.pow(2) + eps1
            if factored:
                s["vr"] = (1 - b) * s["vr"] + b * U.mean(dim=-1)
                s["vc"] = (1 - b) * s["vc"] + b * U.mean(dim=-2)
                u = torch.rsqrt(_approx_sq_grad(s["vr"], s["vc"])) * gg
            else:
                s["v"] = (1 - b) * s["v"] + b * U
                u = torch.rsqrt(s["v"]) * gg
            ru = u.pow(2).mean().sqrt()
            factor = max((ru / clip).item(), 1.0)
            u = u / factor * lr_eff
            s["m"] = b1 * s["m"] + (1 - b1) * u
            if wd != 0:
                p.mul_(1 - lr_eff * wd)
            p.sub_(s["m"])
    return ps


def check_example(solve):
    ps = [torch.tensor([[0.5, -0.2], [0.3, 0.4], [-0.1, 0.2]])]
    grads = [
        [torch.tensor([[0.2, -0.1], [0.1, 0.3], [-0.2, 0.05]])],
        [torch.tensor([[-0.1, 0.25], [0.2, -0.15], [0.05, 0.1]])],
        [torch.tensor([[0.15, -0.05], [-0.2, 0.1], [0.1, -0.3]])],
    ]
    opt = solve(ps, lr=0.1, betas=(0.9, 0.999), scale_parameter=True, weight_decay=0.0)
    _step_all(opt, grads)
    exp = torch.tensor([[0.48983773589134216, -0.202573761343956],
                        [0.2923796474933624, 0.3921566903591156],
                        [-0.09333783388137817, 0.19679373502731323]])
    assert torch.allclose(ps[0], exp, atol=1e-5, rtol=1e-5), f"got {ps[0].tolist()}"


check_example.example = True


def check_matches_oracle_matrix(solve):
    torch.manual_seed(42)
    p0 = [torch.randn(4, 6)]
    gseq = [[torch.randn(4, 6) * 0.3] for _ in range(18)]
    ps = [p.clone() for p in p0]
    opt = solve(ps, lr=1e-2, betas=(0.9, 0.999), weight_decay=0.01, scale_parameter=True)
    _step_all(opt, gseq)
    exp = _oracle_adafactor([p.clone() for p in p0], gseq, 1e-2, (0.9, 0.999), 1e-30, 1e-3, -0.8, 1.0, True, 0.01)
    assert torch.allclose(ps[0], exp[0], atol=1e-5, rtol=1e-5), f"偏离参照:\n{ps[0]}\nvs\n{exp[0]}"


def check_matches_oracle_with_vector(solve):
    # 混合：2D 矩阵走因子化、1D 偏置走整块 EMA
    torch.manual_seed(7)
    p0 = [torch.randn(5, 3), torch.randn(5)]
    gseq = [[torch.randn_like(p) for p in p0] for _ in range(15)]
    ps = [p.clone() for p in p0]
    opt = solve(ps, lr=2e-2, betas=(0.9, 0.999), weight_decay=0.02, scale_parameter=True)
    _step_all(opt, gseq)
    exp = _oracle_adafactor([p.clone() for p in p0], gseq, 2e-2, (0.9, 0.999), 1e-30, 1e-3, -0.8, 1.0, True, 0.02)
    for a, b in zip(ps, exp):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-5)


def check_no_scale_parameter(solve):
    torch.manual_seed(3)
    p0 = [torch.randn(3, 4)]
    gseq = [[torch.randn(3, 4) * 0.2] for _ in range(12)]
    ps = [p.clone() for p in p0]
    opt = solve(ps, lr=1e-2, scale_parameter=False, weight_decay=0.0)
    _step_all(opt, gseq)
    exp = _oracle_adafactor([p.clone() for p in p0], gseq, 1e-2, (0.9, 0.999), 1e-30, 1e-3, -0.8, 1.0, False, 0.0)
    assert torch.allclose(ps[0], exp[0], atol=1e-5, rtol=1e-5)


def check_none_grad_skipped(solve):
    ps = [torch.tensor([[1.0, 2.0]]), torch.tensor([3.0, 4.0])]
    opt = solve(ps, lr=0.1, weight_decay=0.05)
    opt.param_groups[0]["params"][0].grad = torch.tensor([[0.1, -0.1]])
    opt.step()
    assert torch.equal(ps[1], torch.tensor([3.0, 4.0])), "无梯度参数不应被改动"


def check_returns_optimizer_instance(solve):
    opt = solve([torch.randn(2, 2)], lr=0.01)
    assert isinstance(opt, torch.optim.Optimizer)
