import math

import torch


def _step_all(opt, grads_seq):
    """把每步梯度挂到参数对象上再 step；grads_seq[k] 与参数表等长，元素可为 None（无梯度）。"""
    params = opt.param_groups[0]["params"]
    for gs in grads_seq:
        for p, g in zip(params, gs):
            p.grad = g
        opt.step()


def _oracle_galore(params0, proj, grads_seq, lr, betas, eps, wd):
    """独立重算（结构不同）：纯 Python float64 显式双层循环逐项做 Pᵀg / P·û 与逐元素矩更新。"""
    flats = [p.reshape(-1).tolist() for p in params0]
    states = []
    for p0, P in zip(params0, proj):
        r = p0.numel() if P is None else int(P.shape[1])
        states.append({"m": [0.0] * r, "v": [0.0] * r, "t": 0})
    b1, b2 = betas
    for gs in grads_seq:
        for idx, (pl, st, P) in enumerate(zip(flats, states, proj)):
            if gs[idx] is None:
                continue
            g = gs[idx].reshape(-1).tolist()
            r = len(st["m"])
            st["t"] += 1
            t = st["t"]
            if wd != 0:
                for i in range(len(pl)):
                    pl[i] = pl[i] * (1 - lr * wd)
            if P is None:
                gp = list(g)
            else:
                Pd = P.tolist()
                gp = []
                for j in range(r):
                    acc = 0.0
                    for i in range(len(g)):
                        acc += Pd[i][j] * g[i]
                    gp.append(acc)
            for j in range(r):
                st["m"][j] = b1 * st["m"][j] + (1 - b1) * gp[j]
                st["v"][j] = b2 * st["v"][j] + (1 - b2) * gp[j] * gp[j]
            bc1 = 1 - b1 ** t
            bc2 = 1 - b2 ** t
            u = []
            for j in range(r):
                u.append((st["m"][j] / bc1) / (math.sqrt(st["v"][j]) / math.sqrt(bc2) + eps))
            if P is None:
                d = u
            else:
                Pd = P.tolist()
                d = []
                for i in range(len(g)):
                    acc = 0.0
                    for j in range(r):
                        acc += Pd[i][j] * u[j]
                    d.append(acc)
            for i in range(len(pl)):
                pl[i] = pl[i] - lr * d[i]
    return [torch.tensor(fl, dtype=torch.float32).reshape(p.shape) for fl, p in zip(flats, params0)]


def _pattern_grads(n_steps, shapes):
    """确定性梯度图案（非随机）：整型伪随机公式，正负皆有。"""
    seq = []
    for k in range(n_steps):
        gs = []
        for si, sh in enumerate(shapes):
            numel = sh[0] * sh[1] if len(sh) == 2 else sh[0]
            vals = [(((i * 7 + k * 13 + si * 5) % 9) - 4) / 10.0 for i in range(numel)]
            gs.append(torch.tensor(vals, dtype=torch.float32).reshape(sh))
        seq.append(gs)
    return seq


def check_example(solve):
    p = torch.tensor([[1.0, -2.0], [0.5, 0.0]])
    s = 1.0 / math.sqrt(2.0)
    P = torch.tensor([[s, 0.0], [0.0, s], [s, 0.0], [0.0, -s]])
    grads = [
        [torch.tensor([[0.10, -0.20], [0.30, 0.10]])],
        [torch.tensor([[-0.40, 0.05], [0.20, -0.10]])],
        [torch.tensor([[0.15, -0.10], [-0.25, 0.30]])],
    ]
    ps = [p.clone()]
    opt = solve(ps, [P], lr=0.1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, update_proj_gap=2)
    _step_all(opt, grads)
    exp = torch.tensor([[0.9032788872718811, -1.8606481552124023],
                        [0.40477731823921204, -0.13335785269737244]])
    assert torch.allclose(ps[0], exp, atol=1e-5, rtol=1e-5), f"got {ps[0].tolist()}"
    ledger = torch.tensor(opt.state[ps[0]]["recompute_steps"])
    assert torch.equal(ledger, torch.tensor([1, 3])), f"基重算账本应为 [1,3]，得 {ledger.tolist()}"


check_example.example = True


def check_matches_oracle_mixed_proj(solve):
    p0 = [torch.tensor([[0.5, -1.0], [2.0, 0.25], [-0.75, 1.5]]), torch.tensor([1.0, -2.0, 0.5])]
    P = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
                      [1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]]) / 2.0
    proj = [P, None]
    lr, betas, eps, wd, gap = 5e-3, (0.9, 0.98), 1e-8, 0.02, 4
    gseq = _pattern_grads(12, [(3, 2), (3,)])
    ps = [x.clone() for x in p0]
    opt = solve(ps, proj, lr=lr, betas=betas, eps=eps, weight_decay=wd, update_proj_gap=gap)
    _step_all(opt, gseq)
    ref = _oracle_galore([x.clone() for x in p0], proj, gseq, lr, betas, eps, wd)
    for a, b in zip(ps, ref):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-5), f"偏离独立重算:\n{a}\nvs\n{b}"
    ledger = torch.tensor(opt.state[ps[0]]["recompute_steps"])
    expect = torch.tensor([t for t in range(1, 13) if (t - 1) % gap == 0])
    assert torch.equal(ledger, expect), f"账本应为 {expect.tolist()}，得 {ledger.tolist()}"
    assert not ("recompute_steps" in opt.state[ps[1]]), "非投影参数不应有账本"


def check_identity_projection_matches_native_adamw(solve):
    """外部硬参照：P=I（r=N）时与原生 torch.optim.AdamW 逐步一致。"""
    torch.manual_seed(5)
    p0 = torch.randn(2, 3)
    I = torch.eye(6, dtype=torch.float32)
    gseq = [[torch.randn(2, 3)] for _ in range(15)]
    ps = [p0.clone()]
    opt = solve(ps, [I], lr=2e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.05, update_proj_gap=7)
    _step_all(opt, gseq)
    pn = torch.nn.Parameter(p0.clone())
    native = torch.optim.AdamW([pn], lr=2e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.05,
                               foreach=False, fused=False)
    for gs in gseq:
        pn.grad = gs[0]
        native.step()
    assert torch.allclose(ps[0], pn.detach(), atol=1e-5, rtol=1e-5), \
        f"P=I 时须等同 AdamW: {ps[0].tolist()} vs {pn.detach().tolist()}"


def check_none_grad_and_unprojected_skip(solve):
    ps = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    opt = solve(ps, [None, None], lr=0.1, weight_decay=0.1)
    opt.param_groups[0]["params"][0].grad = torch.tensor([0.1, -0.1])
    opt.step()  # 第二个参数无梯度：连权重衰减都不施加
    assert torch.equal(ps[1], torch.tensor([3.0, 4.0])), f"无梯度参数被改动: {ps[1].tolist()}"


def check_gap_one_recomputes_every_step(solve):
    p = torch.tensor([0.3, -0.7, 1.1, 0.2])
    P = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 1.0]])
    gseq = [[torch.tensor([0.2, -0.4, 0.1, 0.3])] for _ in range(6)]
    ps = [p.clone()]
    opt = solve(ps, [P], lr=0.05, update_proj_gap=1)
    _step_all(opt, gseq)
    ledger = torch.tensor(opt.state[ps[0]]["recompute_steps"])
    assert torch.equal(ledger, torch.arange(1, 7)), f"gap=1 应每步重算且不重复记 t=1，得 {ledger.tolist()}"
    assert all(bool(opt.recompute_at(t)) is True for t in range(1, 7)), "recompute_at 与 gap=1 日程矛盾"


def check_zero_rank_only_weight_decay(solve):
    p = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    P = torch.zeros(4, 0)
    gseq = [[torch.tensor([0.2, -0.4, 0.1, 0.3]).reshape(2, 2)] for _ in range(3)]
    ps = [p.clone()]
    opt = solve(ps, [P], lr=0.1, weight_decay=0.0, update_proj_gap=2)
    _step_all(opt, gseq)
    assert torch.equal(ps[0], p), f"r=0 且 wd=0 时参数应严格不变，得 {ps[0].tolist()}"
    ps2 = [p.clone()]
    opt2 = solve(ps2, [P], lr=0.1, weight_decay=0.05, update_proj_gap=2)
    _step_all(opt2, gseq)
    expect = p.clone()
    for _ in range(3):
        expect.mul_(1 - 0.1 * 0.05)  # 只剩解耦权重衰减
    assert torch.allclose(ps2[0], expect, atol=1e-5, rtol=1e-5), f"r=0 时应只剩权重衰减，得 {ps2[0].tolist()}"
    ledger = torch.tensor(opt.state[ps[0]]["recompute_steps"])
    assert torch.equal(ledger, torch.tensor([1, 3])), f"r=0 也须记账，得 {ledger.tolist()}"


def check_basis_sign_flip_invariance(solve):
    """正交基列符号翻转不改变参数轨迹：g̃=Pᵀg 与 Δ=P·û 的翻号互相抵消。"""
    p = torch.tensor([[1.0, -2.0], [0.5, 0.0]])
    s = 1.0 / math.sqrt(2.0)
    P = torch.tensor([[s, 0.0], [0.0, s], [s, 0.0], [0.0, -s]])
    Pflip = P * torch.tensor([-1.0, 1.0])
    gseq = [
        [torch.tensor([[0.10, -0.20], [0.30, 0.10]])],
        [torch.tensor([[-0.40, 0.05], [0.20, -0.10]])],
        [torch.tensor([[0.15, -0.10], [-0.25, 0.30]])],
        [torch.tensor([[0.05, 0.20], [-0.15, -0.10]])],
    ]
    a = [p.clone()]
    b = [p.clone()]
    _step_all(solve(a, [P], lr=0.1, weight_decay=0.01, update_proj_gap=3), gseq)
    _step_all(solve(b, [Pflip], lr=0.1, weight_decay=0.01, update_proj_gap=3), gseq)
    assert torch.equal(a[0], b[0]), f"列翻号应轨迹不变: {a[0].tolist()} vs {b[0].tolist()}"


def check_recompute_at_pure_schedule(solve):
    opt = solve([torch.tensor([1.0])], [None], lr=0.1, update_proj_gap=5)
    flags = [opt.recompute_at(t) for t in range(1, 16)]
    expect = [(t - 1) % 5 == 0 for t in range(1, 16)]
    assert flags == expect, f"recompute_at 日程不符: {flags}"


def check_returns_optimizer_instance(solve):
    P = torch.eye(2)
    opt = solve([torch.tensor([1.0, 2.0])], [P], lr=0.1)
    assert isinstance(opt, torch.optim.Optimizer), "solve 须返回 torch.optim.Optimizer 实例"
