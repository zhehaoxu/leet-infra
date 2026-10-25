import math

import torch


def _step_all(opt, grads_seq):
    for gs in grads_seq:
        for p, g in zip(opt.param_groups[0]["params"], gs):
            p.grad = g
        opt.step()


def _oracle_copt(params, grads_seq, lr, beta1, beta2, eps, wd, rho):
    """独立重算：纯 Python 逐元素 float64 标量循环，与向量化解答结构不同。"""
    ps = [[float(x) for x in p.flatten().tolist()] for p in params]
    states = [dict(m=None, v=None, t=0) for _ in ps]
    for gs in grads_seq:
        for i, gg in enumerate(gs):
            g = [float(x) for x in gg.flatten().tolist()]
            n = len(g)
            s = states[i]
            if s["m"] is None:
                s["m"] = [0.0] * n
                s["v"] = [0.0] * n
            s["t"] += 1
            t = s["t"]
            if wd != 0:
                ps[i] = [x * (1.0 - lr * wd) for x in ps[i]]
            for k in range(n):
                s["m"][k] = beta1 * s["m"][k] + (1.0 - beta1) * g[k]
                s["v"][k] = beta2 * s["v"][k] + (1.0 - beta2) * g[k] * g[k]
            dot = sum(g[k] * s["m"][k] for k in range(n))
            gn = math.sqrt(sum(x * x for x in g))
            mn = math.sqrt(sum(x * x for x in s["m"]))
            cos = dot / (gn * mn) if gn > 0.0 and mn > 0.0 else 1.0
            gate = min(max((cos + 1.0) / 2.0, 0.0), 1.0)
            bc1 = 1.0 - beta1 ** t
            bc2 = math.sqrt(1.0 - beta2 ** t)
            for k in range(n):
                u = (s["m"][k] / bc1) / (math.sqrt(s["v"][k]) / bc2 + eps)
                u = u * gate
                u = max(-rho, min(rho, u))
                ps[i][k] -= lr * u
    return [torch.tensor(flat, dtype=torch.float32).reshape(p.shape) for flat, p in zip(ps, params)]


def check_example(solve):
    ps = [torch.tensor([[0.5, -0.2, 0.1], [0.3, 0.4, -0.1]]), torch.tensor([1.0, -0.5, 0.2])]
    grads = [
        [torch.tensor([[0.2, -0.1, 0.05], [0.1, 0.3, -0.2]]), torch.tensor([0.1, -0.2, 0.05])],
        [torch.tensor([[-0.15, 0.2, 0.1], [0.05, -0.1, 0.3]]), torch.tensor([-0.05, 0.1, 0.2])],
        [torch.tensor([[0.1, 0.05, -0.2], [-0.3, 0.15, 0.1]]), torch.tensor([0.2, -0.1, -0.1])],
    ]
    opt = solve(ps, lr=0.1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, rho=0.5)
    _step_all(opt, grads)
    exp_w = torch.tensor(EXAMPLE_W)
    exp_b = torch.tensor(EXAMPLE_B)
    assert torch.allclose(ps[0], exp_w, atol=1e-5, rtol=1e-5), f"矩阵参数不符: got {ps[0].tolist()}"
    assert torch.allclose(ps[1], exp_b, atol=1e-5, rtol=1e-5), f"一维参数不符: got {ps[1].tolist()}"
    ref = _oracle_copt(
        [torch.tensor(EXAMPLE_P0_T), torch.tensor(EXAMPLE_P1_T)],
        grads, 0.1, 0.9, 0.999, 1e-8, 0.01, 0.5,
    )
    assert torch.allclose(ps[0], ref[0], atol=1e-5, rtol=1e-5), "示例与独立重算不一致"
    assert torch.allclose(ps[1], ref[1], atol=1e-5, rtol=1e-5), "示例与独立重算不一致"


check_example.example = True

EXAMPLE_P0_T = [[0.5, -0.2, 0.1], [0.3, 0.4, -0.1]]
EXAMPLE_P1_T = [1.0, -0.5, 0.2]
EXAMPLE_W = [[0.4150408804, -0.2133053988, 0.0172756668], [0.2304721773, 0.270242095, -0.0996336192]]
EXAMPLE_B = [0.8751174808, -0.3889863789, 0.0713768303]


def check_matches_oracle_random(solve):
    torch.manual_seed(7)
    p0 = [torch.randn(4, 3), torch.randn(6)]
    gseq = [[torch.randn_like(p) * 0.4 for p in p0] for _ in range(12)]
    ps = [p.clone() for p in p0]
    opt = solve(ps, lr=3e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.05, rho=0.3)
    _step_all(opt, gseq)
    exp = _oracle_copt([p.clone() for p in p0], gseq, 3e-2, 0.9, 0.999, 1e-8, 0.05, 0.3)
    for a, b in zip(ps, exp):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-5), f"偏离独立重算:\n{a}\nvs\n{b}"


def check_aligned_equals_adamw(solve):
    """恒定方向的梯度 ⇒ cos=1、gate=1，且 |u| < ρ 时整条轨迹与原生 AdamW 逐位一致。"""
    torch.manual_seed(11)
    p0 = [torch.randn(3, 3) * 0.5]
    gseq = []
    base = torch.randn(3, 3) * 0.2
    for _ in range(10):
        gseq.append([base.clone()])
    kw = dict(lr=0.05, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, rho=1e9)
    ps = [p.clone() for p in p0]
    _step_all(solve(ps, **kw), gseq)
    ref = [p.clone() for p in p0]
    adamw = torch.optim.AdamW(ref, lr=0.05, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    for gs in gseq:
        for p, g in zip(ref, gs):
            p.grad = g
        adamw.step()
    assert torch.allclose(ps[0], ref[0], atol=1e-6, rtol=1e-6), f"应与 AdamW 一致:\n{ps[0]}\nvs\n{ref[0]}"


def check_signflip_gated_out(solve):
    """长时间同向后突然反向：动量尚未掉头，cos=−1 ⇒ gate=0，该步参数几乎不动。"""
    p0 = torch.tensor([1.0, 1.0])
    ps = [p0.clone()]
    g = torch.tensor([0.3, -0.1])
    opt = solve(ps, lr=0.1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, rho=1e9)
    params = opt.param_groups[0]["params"]
    for _ in range(20):
        params[0].grad = g.clone()
        opt.step()
    before = params[0].clone()
    params[0].grad = (-0.5 * g).clone()
    opt.step()
    delta = (params[0] - before).abs().max().item()
    assert delta < 1e-7, f"反向步应被连续门压死，实得位移 {delta}"


def check_step_bounded_by_lr_rho(solve):
    """不变量：无权重衰减时每步 |Δp| ≤ lr·ρ（截幅承担限幅）。"""
    torch.manual_seed(3)
    ps = [torch.randn(5)]
    opt = solve(ps, lr=0.03, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, rho=0.1)
    params = opt.param_groups[0]["params"]
    for i in range(8):
        params[0].grad = torch.randn(5) * 7.0
        b = params[0].clone()
        opt.step()
        d = (params[0] - b).abs().max().item()
        assert d <= 0.03 * 0.1 + 1e-6, f"第 {i} 步位移 {d} 超出 lr·rho"


def check_none_grad_param_skipped(solve):
    ps = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    opt = solve(ps, lr=0.1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1, rho=1.0)
    params = opt.param_groups[0]["params"]
    params[0].grad = torch.tensor([0.1, -0.1])
    opt.step()
    assert torch.equal(ps[1], torch.tensor([3.0, 4.0])), f"无梯度参数被改动: {ps[1].tolist()}"


def check_returns_optimizer_instance(solve):
    opt = solve([torch.tensor([1.0, 2.0])], lr=0.1)
    assert isinstance(opt, torch.optim.Optimizer), "solve 须返回 torch.optim.Optimizer 实例"
