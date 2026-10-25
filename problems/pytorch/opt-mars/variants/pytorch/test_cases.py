import math

import torch


def _step_all(opt, grads_seq):
    """按固定梯度序列驱动优化器：每步把梯度挂到参数对象上再 step()。"""
    for gs in grads_seq:
        for p, g in zip(opt.param_groups[0]["params"], gs):
            p.grad = g
        opt.step()


def _oracle_mars(params, grads_seq, lr, beta1, beta2, gamma, eps, wd, ams):
    """独立参照：纯 Python 逐张量、逐元素显式循环（float64 标量运算），与向量化解答结构不同。"""
    ps = [[float(x) for x in p.flatten().tolist()] for p in params]
    states = [dict(m=None, v=None, last=None, vmax=None, t=0) for _ in ps]
    gp = gamma * beta1 / (1.0 - beta1)
    for gs in grads_seq:
        for i, gg in enumerate(gs):
            g = [float(x) for x in gg.flatten().tolist()]
            n = len(g)
            s = states[i]
            if s["m"] is None:
                s["m"] = [0.0] * n
                s["v"] = [0.0] * n
                s["last"] = [0.0] * n  # 首步约定 g_0 := 0
                s["vmax"] = [0.0] * n
            s["t"] += 1
            t = s["t"]
            if wd != 0:
                ps[i] = [x * (1.0 - lr * wd) for x in ps[i]]
            c = [g[k] + gp * (g[k] - s["last"][k]) for k in range(n)]
            nrm = math.sqrt(sum(x * x for x in c))
            if nrm > 1.0:
                c = [x / nrm for x in c]
            bc1 = 1.0 - beta1 ** t
            bc2s = math.sqrt(1.0 - beta2 ** t)
            for k in range(n):
                s["m"][k] = beta1 * s["m"][k] + (1.0 - beta1) * c[k]
                s["v"][k] = beta2 * s["v"][k] + (1.0 - beta2) * c[k] * c[k]
                base = s["v"][k]
                if ams:
                    s["vmax"][k] = max(s["vmax"][k], s["v"][k])
                    base = s["vmax"][k]
                denom = math.sqrt(base) / bc2s + eps
                ps[i][k] -= (lr / bc1) * s["m"][k] / denom
            s["last"] = list(g)
    return [torch.tensor(flat, dtype=torch.float32).reshape(p.shape) for flat, p in zip(ps, params)]


def check_example(solve):
    ps = [torch.tensor([[0.5, -0.2, 0.1], [0.3, 0.4, -0.1]]), torch.tensor([1.0, -0.5, 0.2])]
    grads = [
        [torch.tensor([[0.2, -0.1, 0.05], [0.1, 0.3, -0.2]]), torch.tensor([0.1, -0.2, 0.05])],
        [torch.tensor([[-0.15, 0.2, 0.1], [0.05, -0.1, 0.3]]), torch.tensor([-0.05, 0.1, 0.2])],
        [torch.tensor([[0.1, 0.05, -0.2], [-0.3, 0.15, 0.1]]), torch.tensor([0.2, -0.1, -0.1])],
    ]
    opt = solve(ps, lr=0.1, betas=(0.9, 0.99), gamma=0.025, eps=1e-8, weight_decay=0.01, ams_bound=False)
    _step_all(opt, grads)
    exp_w = torch.tensor(EXAMPLE_W)
    exp_b = torch.tensor(EXAMPLE_B)
    assert torch.allclose(ps[0], exp_w, atol=1e-5, rtol=1e-5), f"矩阵参数不符: got {ps[0].tolist()}"
    assert torch.allclose(ps[1], exp_b, atol=1e-5, rtol=1e-5), f"一维参数不符: got {ps[1].tolist()}"
    ref = _oracle_mars([p.clone() for p in [torch.tensor(EXAMPLE_P0_T), torch.tensor(EXAMPLE_P1_T)]],
                       grads, 0.1, 0.9, 0.99, 0.025, 1e-8, 0.01, False)
    assert torch.allclose(ps[0], ref[0], atol=1e-5, rtol=1e-5), "示例与独立重算不一致"


check_example.example = True

EXAMPLE_P0_T = [[0.5, -0.2, 0.1], [0.3, 0.4, -0.1]]
EXAMPLE_P1_T = [1.0, -0.5, 0.2]
EXAMPLE_W = [[0.3747776747, -0.1735291779, -0.0720640868], [0.1518298835, 0.2275365144, -0.0598083213]]
EXAMPLE_B = [0.8236305714, -0.3480571508, -0.0091746142]


def check_matches_oracle_random(solve):
    torch.manual_seed(42)
    p0 = [torch.randn(3, 4), torch.randn(5)]
    gseq = [[torch.randn_like(p) * 0.3 for p in p0] for _ in range(15)]
    ps = [p.clone() for p in p0]
    opt = solve(ps, lr=2e-2, betas=(0.9, 0.99), gamma=0.05, eps=1e-8, weight_decay=0.05, ams_bound=False)
    _step_all(opt, gseq)
    exp = _oracle_mars([p.clone() for p in p0], gseq, 2e-2, 0.9, 0.99, 0.05, 1e-8, 0.05, False)
    for a, b in zip(ps, exp):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-5), f"偏离独立重算:\n{a}\nvs\n{b}"


def check_gamma_zero_reduces_to_adamw(solve):
    """gamma=0 ⇒ c_t ≡ g_t；梯度范数全程 <1 ⇒ 信任比不触发 ⇒ 与原生 AdamW 逐位一致。"""
    torch.manual_seed(7)
    p0 = [torch.randn(4, 2), torch.randn(6)]
    gseq = [[torch.randn_like(p) * 0.05 for p in p0] for _ in range(12)]
    ps = [p.clone() for p in p0]
    opt = solve(ps, lr=5e-3, betas=(0.9, 0.99), gamma=0.0, eps=1e-8, weight_decay=0.02, ams_bound=False)
    _step_all(opt, gseq)
    ref_ps = [p.clone() for p in p0]
    adamw = torch.optim.AdamW(ref_ps, lr=5e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.02,
                              foreach=False, fused=False)
    for gs in gseq:
        for p, g in zip(ref_ps, gs):
            p.grad = g
        adamw.step()
    for a, b in zip(ps, ref_ps):
        assert torch.allclose(a, b, atol=1e-7, rtol=1e-7), "gamma=0 应退化为 AdamW"


def check_trust_clip_bounds_first_moment(solve):
    """‖c‖_F > 1 触发裁剪后 ‖c̃‖=1，首步 m₁=(1−β1)c̃ 的范数恰为 1−β1，与梯度尺度无关。"""
    g = torch.randn(3, 3) * 10.0
    ps = [torch.zeros(3, 3)]
    opt = solve(ps, lr=0.1, betas=(0.9, 0.99), gamma=0.025, eps=1e-8, weight_decay=0.0)
    opt.param_groups[0]["params"][0].grad = g
    opt.step()
    m1 = opt.state[ps[0]]["exp_avg"]
    assert abs(float(torch.linalg.vector_norm(m1)) - 0.1) < 1e-5, f"裁剪后 ‖m1‖ 应为 0.1，得 {float(torch.linalg.vector_norm(m1))}"

    ps2 = [torch.zeros(3, 3)]
    opt2 = solve(ps2, lr=0.1, betas=(0.9, 0.99), gamma=0.025, eps=1e-8, weight_decay=0.0)
    opt2.param_groups[0]["params"][0].grad = g * 100.0  # 放大百倍，裁剪后仍应同范数
    opt2.step()
    m1b = opt2.state[ps2[0]]["exp_avg"]
    assert abs(float(torch.linalg.vector_norm(m1b)) - 0.1) < 1e-4, "norm-clip 应使更新界与梯度尺度解耦"


def check_ams_bound_running_max_bitwise(solve):
    """尖峰后转平的梯度序列：v 衰减时 v_max 钉住历史峰值；两种开关结果必须不同。"""
    torch.manual_seed(3)
    p0 = [torch.randn(4, 4) * 0.5]
    big = torch.randn(4, 4) * 0.25   # ‖c‖<1，避免触发 clip，隔离 ams 效果
    small = torch.randn(4, 4) * 0.01
    gseq = [[big]] + [[small * (0.9 ** k)] for k in range(10)]
    outs = []
    for ams in (False, True):
        ps = [p.clone() for p in p0]
        opt = solve(ps, lr=2e-2, betas=(0.9, 0.99), gamma=0.025, eps=1e-8, weight_decay=0.0, ams_bound=ams)
        _step_all(opt, gseq)
        exp = _oracle_mars([p.clone() for p in p0], gseq, 2e-2, 0.9, 0.99, 0.025, 1e-8, 0.0, ams)
        assert torch.allclose(ps[0], exp[0], atol=1e-5, rtol=1e-5), "ams_bound 与独立重算不符"
        outs.append(ps[0].clone())
    d = float((outs[0] - outs[1]).abs().max())
    assert d > 1e-6, "尖峰-衰减序列下 ams_bound 必须与不分界情形可区分"


def check_zero_grad_noop(solve):
    p0 = torch.tensor([[0.3, -0.7], [0.1, 0.5]])
    ps = [p0.clone()]
    opt = solve(ps, lr=0.1, betas=(0.9, 0.99), gamma=0.025, eps=1e-8, weight_decay=0.0)
    for _ in range(5):
        opt.param_groups[0]["params"][0].grad = torch.zeros(2, 2)
        opt.step()
    assert torch.equal(ps[0], p0), f"零梯度且无权重衰减时参数应原样不变: {ps[0].tolist()}"


def check_none_grad_param_skipped(solve):
    ps = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    opt = solve(ps, lr=0.1, betas=(0.9, 0.99), gamma=0.025, eps=1e-8, weight_decay=0.1)
    opt.param_groups[0]["params"][0].grad = torch.tensor([0.1, -0.1])
    opt.step()
    assert torch.equal(ps[1], torch.tensor([3.0, 4.0])), f"无梯度参数被改动（含权重衰减）: {ps[1].tolist()}"


def check_returns_optimizer_instance(solve):
    opt = solve([torch.randn(3, 3)], lr=1e-3)
    assert isinstance(opt, torch.optim.Optimizer), "solve 须返回 torch.optim.Optimizer 实例"
