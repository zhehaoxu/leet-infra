import math

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


def _oracle_adamuon(params, grads_seq, lr, mu, ns_steps, beta1, beta2, eps, wd):
    """独立参照：不走 Optimizer 类，状态用 dict、2-D 用函数式重算、1-D 用 float64 标量循环。"""
    ps = [p.clone() for p in params]
    m = [torch.zeros_like(p) for p in ps]
    v = [torch.zeros_like(p) for p in ps]
    t = 0
    for gs in grads_seq:
        t += 1
        for i, gg in enumerate(gs):
            if wd != 0:
                ps[i].mul_(1 - lr * wd)
            if ps[i].dim() >= 2:
                m[i] = mu * m[i] + (1 - mu) * gg
                G = (1 - mu) * gg + mu * m[i]
                O = _ns(G, ns_steps)
                v[i] = beta2 * v[i] + (1 - beta2) * O * O
                vhat = v[i] / (1 - beta2 ** t)
                U = O / (vhat.sqrt() + eps)
                U = U * (O.norm() / (U.norm() + 1e-12))
                ps[i].sub_(lr * U.reshape(ps[i].shape))
            else:
                g = [float(x) for x in gg.flatten().tolist()]
                mv = [float(x) for x in m[i].flatten().tolist()]
                vv = [float(x) for x in v[i].flatten().tolist()]
                bc1 = 1.0 - beta1 ** t
                bc2 = math.sqrt(1.0 - beta2 ** t)
                out = []
                for k in range(len(g)):
                    mv[k] = beta1 * mv[k] + (1.0 - beta1) * g[k]
                    vv[k] = beta2 * vv[k] + (1.0 - beta2) * g[k] * g[k]
                    out.append(float(ps[i].flatten()[k]) - lr * (mv[k] / bc1) / (math.sqrt(vv[k]) / bc2 + eps))
                ps[i] = torch.tensor(out, dtype=torch.float32).reshape(ps[i].shape)
                m[i] = torch.tensor(mv, dtype=torch.float32).reshape(m[i].shape)
                v[i] = torch.tensor(vv, dtype=torch.float32).reshape(v[i].shape)
    return ps


def check_example(solve):
    ps = [torch.tensor([[0.5, -0.2, 0.1], [0.3, 0.4, -0.1]]), torch.tensor([1.0, -0.5, 0.2])]
    grads = [
        [torch.tensor([[0.2, -0.1, 0.05], [0.1, 0.3, -0.2]]), torch.tensor([0.1, -0.2, 0.05])],
        [torch.tensor([[-0.15, 0.2, 0.1], [0.05, -0.1, 0.3]]), torch.tensor([-0.05, 0.1, 0.2])],
        [torch.tensor([[0.1, 0.05, -0.2], [-0.3, 0.15, 0.1]]), torch.tensor([0.2, -0.1, -0.1])],
    ]
    opt = solve(ps, lr=0.1, momentum=0.95, ns_steps=5, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)
    _step_all(opt, grads)
    exp_w = torch.tensor(EXAMPLE_W)
    exp_b = torch.tensor(EXAMPLE_B)
    assert torch.allclose(ps[0], exp_w, atol=1e-5, rtol=1e-5), f"矩阵参数不符: got {ps[0].tolist()}"
    assert torch.allclose(ps[1], exp_b, atol=1e-5, rtol=1e-5), f"一维参数不符: got {ps[1].tolist()}"
    ref = _oracle_adamuon(
        [torch.tensor(EXAMPLE_P0_T), torch.tensor(EXAMPLE_P1_T)],
        grads, 0.1, 0.95, 5, 0.9, 0.95, 1e-8, 0.01,
    )
    assert torch.allclose(ps[0], ref[0], atol=1e-5, rtol=1e-5), "示例与独立重算不一致"
    assert torch.allclose(ps[1], ref[1], atol=1e-5, rtol=1e-5), "示例与独立重算不一致"


check_example.example = True

EXAMPLE_P0_T = [[0.5, -0.2, 0.1], [0.3, 0.4, -0.1]]
EXAMPLE_P1_T = [1.0, -0.5, 0.2]
EXAMPLE_W = [
    [0.4479453564, -0.2746757567, 0.0711846948],
    [0.2389980704, 0.3081416488, -0.1262495071],
]
EXAMPLE_B = [0.8055183291, -0.3264307976, -0.0212844685]


def check_matches_oracle_mixed(solve):
    torch.manual_seed(42)
    p0 = [torch.randn(5, 2), torch.randn(3, 4), torch.randn(6)]
    gseq = [[torch.randn_like(p) * 0.3 for p in p0] for _ in range(12)]
    ps = [p.clone() for p in p0]
    opt = solve(ps, lr=3e-2, momentum=0.9, ns_steps=5, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.05)
    _step_all(opt, gseq)
    exp = _oracle_adamuon([p.clone() for p in p0], gseq, 3e-2, 0.9, 5, 0.9, 0.95, 1e-8, 0.05)
    for a, b in zip(ps, exp):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-5), f"偏离独立重算:\n{a}\nvs\n{b}"


def check_one_dim_is_adamw(solve):
    """1-D 参数走纯 AdamW 路径：与原生 torch.optim.AdamW 轨迹逐位一致。"""
    torch.manual_seed(7)
    p0 = [torch.randn(4)]
    gseq = [[torch.randn_like(p) for p in p0] for _ in range(10)]
    kw = dict(lr=0.05, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    ps = [p.clone() for p in p0]
    _step_all(solve(ps, momentum=0.95, ns_steps=5, **kw), gseq)
    ref = [p.clone() for p in p0]
    adamw = torch.optim.AdamW(ref, lr=0.05, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    for gs in gseq:
        for p, g in zip(ref, gs):
            p.grad = g
        adamw.step()
    assert torch.allclose(ps[0], ref[0], atol=1e-6, rtol=1e-6), f"1-D 应等价 AdamW:\n{ps[0]}\nvs\n{ref[0]}"


def check_first_step_sign_plateau(solve):
    """首步 V̂=(1−β2)O² ⇒ √V̂ ∝ |O|，除完逐元素只剩符号，范数还原后 |Δp| 全部相等。"""
    torch.manual_seed(5)
    ps = [torch.randn(4, 3)]
    opt = solve(ps, lr=0.1, momentum=0.9, ns_steps=5, betas=(0.9, 0.95), eps=1e-12, weight_decay=0.0)
    params = opt.param_groups[0]["params"]
    params[0].grad = torch.randn(4, 3) * 0.5
    before = params[0].clone()
    opt.step()
    delta = (params[0] - before).abs()
    flat = delta.flatten()
    assert torch.allclose(flat, flat[0].expand_as(flat), atol=1e-4, rtol=1e-3), \
        f"首步位移应是等幅 sign 板: {flat.tolist()}"
    assert flat[0].item() > 0, "位移不能为 0"


def check_step_norm_equals_lr_norm_o(solve):
    """范数还原不变量：无 wd 时每步 ‖Δp‖_F == lr·‖O_raw‖_F（用独立 NS 复算 ‖O‖）。"""
    torch.manual_seed(13)
    p0 = torch.randn(3, 4)
    ps = [p0.clone()]
    lr = 0.05
    opt = solve(ps, lr=lr, momentum=0.9, ns_steps=5, betas=(0.9, 0.95), eps=1e-12, weight_decay=0.0)
    params = opt.param_groups[0]["params"]
    g = torch.randn(3, 4)
    m_prev = torch.zeros_like(p0)
    params[0].grad = g.clone()
    before = params[0].clone()
    opt.step()
    m_new = 0.9 * m_prev + 0.1 * g
    G = 0.1 * g + 0.9 * m_new
    O = _ns(G, 5)
    expect = lr * O.norm().item()
    got = (params[0] - before).norm().item()
    assert abs(got - expect) <= 1e-4 * max(1.0, expect), f"‖Δp‖={got} 应为 lr·‖O‖={expect}"


def check_none_grad_param_skipped(solve):
    ps = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    opt = solve(ps, lr=0.1, weight_decay=0.1)
    params = opt.param_groups[0]["params"]
    params[0].grad = torch.tensor([0.1, -0.1])
    opt.step()
    assert torch.equal(ps[1], torch.tensor([3.0, 4.0])), f"无梯度参数被改动: {ps[1].tolist()}"


def check_returns_optimizer_instance(solve):
    opt = solve([torch.tensor([1.0, 2.0])], lr=0.1)
    assert isinstance(opt, torch.optim.Optimizer), "solve 须返回 torch.optim.Optimizer 实例"
