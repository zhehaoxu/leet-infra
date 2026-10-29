import math

import torch
import torch.nn.functional as F

LOGPS_W = torch.tensor([[-0.5, -0.4, -0.6], [-0.2, -0.3, -0.1]], dtype=torch.float64)
LOGPS_L = torch.tensor([[-0.7, -0.8, -0.9], [-0.4, -0.5, -0.6]], dtype=torch.float64)
REF_W = torch.tensor([-1.5, -1.0], dtype=torch.float64)
REF_L = torch.tensor([-2.4, -1.5], dtype=torch.float64)
MASK_W = torch.ones(2, 3, dtype=torch.bool)
MASK_L = torch.tensor([[True, True, False], [True, True, True]], dtype=torch.bool)
BETA = 0.1
EXAMPLE = {
    "loss": 0.7062532531268221,
    "grad_w": [[-0.026124241239590007] * 3, [-0.02450006665600173] * 3],
    "grad_l": [[0.026124241239590007, 0.026124241239590007, 0.0], [0.02450006665600173] * 3],
}


def _forward_autograd(w, l, rw, rl, mw, ml, beta):
    s_w = (w * mw).sum(-1)
    s_l = (l * ml).sum(-1)
    delta = (s_w - rw) - (s_l - rl)
    return F.softplus(-beta * delta).mean()


def check_example(solve):
    out = solve(LOGPS_W, LOGPS_L, REF_W, REF_L, MASK_W, MASK_L, BETA)
    assert set(out) == {"loss", "grad_logp_w", "grad_logp_l"}, set(out)
    assert abs(out["loss"].item() - EXAMPLE["loss"]) < 1e-12, out["loss"].item()
    assert torch.allclose(out["grad_logp_w"], torch.tensor(EXAMPLE["grad_w"], dtype=torch.float64), atol=1e-12)
    assert torch.allclose(out["grad_logp_l"], torch.tensor(EXAMPLE["grad_l"], dtype=torch.float64), atol=1e-12)
    assert out["loss"].dim() == 0 and out["loss"].dtype == torch.float64
    assert out["grad_logp_w"].shape == (2, 3) and out["grad_logp_w"].dtype == torch.float64
    assert out["grad_logp_l"].dtype == torch.float64


check_example.example = True


def check_matches_autograd(solve):
    torch.manual_seed(91)
    for _ in range(4):
        b, t = 3, 4
        w = torch.randn(b, t, dtype=torch.float64) * 0.5
        l = torch.randn(b, t, dtype=torch.float64) * 0.5
        rw = torch.randn(b, dtype=torch.float64)
        rl = torch.randn(b, dtype=torch.float64)
        mw = torch.rand(b, t) > 0.2
        ml = torch.rand(b, t) > 0.2
        mw[:, 0] = True
        ml[:, 0] = True
        beta = 0.4
        out = solve(w, l, rw, rl, mw, ml, beta)

        w2 = w.clone().requires_grad_(True)
        l2 = l.clone().requires_grad_(True)
        loss_ref = _forward_autograd(w2, l2, rw, rl, mw, ml, beta)
        loss_ref.backward()
        assert abs(out["loss"].item() - loss_ref.item()) < 1e-12, (out["loss"].item(), loss_ref.item())
        assert torch.allclose(out["grad_logp_w"], w2.grad, atol=1e-12), (out["grad_logp_w"], w2.grad)
        assert torch.allclose(out["grad_logp_l"], l2.grad, atol=1e-12), (out["grad_logp_l"], l2.grad)


def check_padding_zero_grad(solve):
    # 被 mask 的位置梯度严格为 0，且改动该位置的 logp 不影响损失与其它位置的梯度
    perturbed = LOGPS_L.clone()
    perturbed[0, 2] = -123.0
    base = solve(LOGPS_W, LOGPS_L, REF_W, REF_L, MASK_W, MASK_L, BETA)
    other = solve(LOGPS_W, perturbed, REF_W, REF_L, MASK_W, MASK_L, BETA)
    assert base["grad_logp_l"][0, 2].item() == 0.0
    assert abs(base["loss"].item() - other["loss"].item()) < 1e-12
    assert torch.equal(base["grad_logp_l"], other["grad_logp_l"])


def check_sign_relation(solve):
    # 同一位置两侧都有效时 grad_l = -grad_w；不同位置的比值由各自 mask 决定
    out = solve(LOGPS_W, LOGPS_L, REF_W, REF_L, MASK_W, MASK_W, BETA)
    both = MASK_W
    assert torch.allclose(out["grad_logp_w"][both], -out["grad_logp_l"][both], atol=1e-12)
    assert torch.all(out["grad_logp_w"] < 0) and torch.all(out["grad_logp_l"] > 0)
    # 第 b 条样本的 dL/dΔ_b = -β·σ(-x_b)/B 对所有有效 token 相同
    assert torch.allclose(out["grad_logp_w"][0], out["grad_logp_w"][0, :1].expand(3), atol=1e-12)


def check_loss_scaling(solve):
    # 两个样本取均值：单样本的梯度是两条同分布样本时的 2 倍
    w = torch.tensor([[-0.5, -0.4]], dtype=torch.float64)
    l = torch.tensor([[-0.7, -0.8]], dtype=torch.float64)
    rw = torch.tensor([-1.5], dtype=torch.float64)
    rl = torch.tensor([-2.4], dtype=torch.float64)
    m = torch.ones(1, 2, dtype=torch.bool)
    one = solve(w, l, rw, rl, m, m, BETA)
    two = solve(w.repeat(2, 1), l.repeat(2, 1), rw.repeat(2), rl.repeat(2), m.repeat(2, 1), m.repeat(2, 1), BETA)
    assert abs(one["loss"].item() - two["loss"].item()) < 1e-12
    assert torch.allclose(one["grad_logp_w"], 2.0 * two["grad_logp_w"], atol=1e-12)


def check_beta_at_zero_margin(solve):
    # Δ = 0 时 x = 0：loss = ln2，dL/dΔ = -β/2（与 β 成正比）
    w = torch.tensor([[-1.0, 0.0]], dtype=torch.float64)
    l = torch.tensor([[-1.0, 0.0]], dtype=torch.float64)
    rw = torch.tensor([-1.0], dtype=torch.float64)
    rl = torch.tensor([-1.0], dtype=torch.float64)
    m = torch.ones(1, 2, dtype=torch.bool)
    for beta in (0.5, 2.0):
        out = solve(w, l, rw, rl, m, m, beta)
        assert abs(out["loss"].item() - math.log(2.0)) < 1e-12, out["loss"].item()
        assert abs(out["grad_logp_w"][0, 1].item() - (-beta / 2.0)) < 1e-12, out["grad_logp_w"]


def check_float32_input(solve):
    out = solve(LOGPS_W.to(torch.float32), LOGPS_L.to(torch.float32), REF_W.to(torch.float32),
                REF_L.to(torch.float32), MASK_W, MASK_L, BETA)
    assert out["loss"].dtype == torch.float64 and out["grad_logp_w"].dtype == torch.float64
    assert abs(out["loss"].item() - EXAMPLE["loss"]) < 1e-6, out["loss"].item()


def check_errors(solve):
    w = torch.zeros(1, 2, dtype=torch.float64)
    m = torch.ones(1, 2, dtype=torch.bool)
    r = torch.zeros(1, dtype=torch.float64)

    def bad(fn, desc):
        try:
            fn()
        except ValueError:
            return
        raise AssertionError(f"应抛 ValueError: {desc}")

    bad(lambda: solve([[0.0, 0.0]], w, r, r, m, m, 0.1), "logp_w 非张量")
    bad(lambda: solve(w[0], w, r, r, m, m, 0.1), "logp_w 非二维")
    bad(lambda: solve(w, torch.zeros(2, 2, dtype=torch.float64), r, r, m, m, 0.1), "logp_l 形状不符")
    bad(lambda: solve(w, w, torch.zeros(2, dtype=torch.float64), r, m, m, 0.1), "ref_w 形状不符")
    bad(lambda: solve(w, w, r.to(torch.float32), r, m, m, 0.1), "ref_w dtype 不符")
    bad(lambda: solve(w, w, r, r, m.to(torch.int64), m, 0.1), "mask_w 非 bool")
    bad(lambda: solve(w, w, r, r, m[:, :1], m, 0.1), "mask_w 形状不符")
    bad(lambda: solve(w, w, r, r, torch.zeros(1, 2, dtype=torch.bool), m, 0.1), "mask_w 有空样本")
    bad(lambda: solve(w, w, r, r, m, m, 0.0), "beta=0")
    bad(lambda: solve(w, w, r, r, m, m, -0.5), "beta<0")
    bad(lambda: solve(torch.full((1, 2), float("nan"), dtype=torch.float64), w, r, r, m, m, 0.1), "含 NaN")
