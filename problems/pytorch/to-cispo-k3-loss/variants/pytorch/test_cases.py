import math

import torch

LOGPS = torch.tensor([[0.1, -0.2, 0.0], [0.3, 0.0, 0.5]], dtype=torch.float64)
LOGPS_OLD = torch.zeros(2, 3, dtype=torch.float64)
REF_LOGPS = torch.tensor([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]], dtype=torch.float64)
MASK = torch.tensor([[True, True, True], [True, True, False]], dtype=torch.bool)
ADV = torch.tensor([1.0, -0.5], dtype=torch.float64)
EXAMPLE = {
    "loss": 0.04664581176160632,
    "kl_k3": 0.006282218854355426,
    "clip_frac": 0.2,
}


def _oracle(lp, old, ref, mask, adv, eps_low, eps_high):
    n = len(lp)
    t_len = len(lp[0])
    count = 0
    total = 0.0
    kl = 0.0
    outside = 0
    for i in range(n):
        for j in range(t_len):
            if not mask[i][j]:
                continue
            count += 1
            rho = math.exp(lp[i][j] - old[i][j])
            w = min(max(rho, 1.0 - eps_low), 1.0 + eps_high)
            total += w * adv[i] * lp[i][j]
            log_r = ref[i][j] - lp[i][j]
            kl += math.exp(log_r) - log_r - 1.0
            if rho < 1.0 - eps_low or rho > 1.0 + eps_high:
                outside += 1
    return -total / count, kl / count, outside / count


def check_example(solve):
    out = solve(LOGPS, LOGPS_OLD, REF_LOGPS, MASK, ADV, 0.2, 0.2)
    assert set(out) == {"loss", "kl_k3", "clip_frac"}, set(out)
    assert abs(out["loss"].item() - EXAMPLE["loss"]) < 1e-12, out["loss"].item()
    assert abs(out["kl_k3"].item() - EXAMPLE["kl_k3"]) < 1e-12, out["kl_k3"].item()
    assert abs(out["clip_frac"].item() - EXAMPLE["clip_frac"]) < 1e-12, out["clip_frac"].item()
    for key in ("loss", "kl_k3", "clip_frac"):
        assert out[key].dim() == 0 and out[key].dtype == torch.float64, key


check_example.example = True


def check_oracle(solve):
    torch.manual_seed(71)
    for _ in range(5):
        n, t = 3, 4
        lp = torch.randn(n, t, dtype=torch.float64) * 0.5
        old = torch.randn(n, t, dtype=torch.float64) * 0.5
        ref = lp + torch.randn(n, t, dtype=torch.float64) * 0.3
        m = torch.rand(n, t) > 0.25
        m[:, 0] = True
        adv = torch.randn(n, dtype=torch.float64)
        out = solve(lp, old, ref, m, adv, 0.1, 0.3)
        e_loss, e_kl, e_frac = _oracle(lp.tolist(), old.tolist(), ref.tolist(), m.tolist(),
                                       adv.tolist(), 0.1, 0.3)
        assert abs(out["loss"].item() - e_loss) < 1e-12, (out["loss"].item(), e_loss)
        assert abs(out["kl_k3"].item() - e_kl) < 1e-12, (out["kl_k3"].item(), e_kl)
        assert abs(out["clip_frac"].item() - e_frac) < 1e-12, (out["clip_frac"].item(), e_frac)


def check_stop_gradient(solve):
    # CISPO：裁剪权重 w 停梯度，dL/dlogp = -w·A/N（有效 token），masked 位置为 0。
    # 若忘了 detach，未裁剪位置会多出 d(rho)/dlogp = rho 的一项，梯度明显不同。
    lp = LOGPS.clone().requires_grad_(True)
    out = solve(lp, LOGPS_OLD, REF_LOGPS, MASK, ADV, 0.2, 0.2)
    out["loss"].backward()
    n_valid = 5.0
    weights = [[math.exp(0.1), math.exp(-0.2), 1.0], [1.2, 1.0, 0.0]]  # clamp 后逐 token 权重
    expected = torch.zeros(2, 3, dtype=torch.float64)
    for i in range(2):
        for j in range(3):
            if MASK[i, j]:
                expected[i, j] = -weights[i][j] * ADV[i] / n_valid
    assert torch.allclose(lp.grad, expected, atol=1e-12), (lp.grad, expected)
    assert lp.grad[1, 2].item() == 0.0


def check_k3_zero_when_on_policy(solve):
    # 参考策略与当前策略相同 -> K3 每个 token 都是 0
    lp = torch.tensor([[0.2, -0.4], [0.0, 0.1]], dtype=torch.float64)
    out = solve(lp, lp.clone(), lp.clone(), torch.ones(2, 2, dtype=torch.bool),
                torch.zeros(2, dtype=torch.float64))
    assert abs(out["kl_k3"].item()) < 1e-12
    assert abs(out["loss"].item()) < 1e-12


def check_k3_formula_direction(solve):
    # log_r = ref - logp（参考减当前）；方向写反会得到不同数值
    lp = torch.tensor([[-1.0]], dtype=torch.float64)
    ref = torch.tensor([[-0.5]], dtype=torch.float64)
    out = solve(lp, lp.clone(), ref, torch.ones(1, 1, dtype=torch.bool),
                torch.zeros(1, dtype=torch.float64))
    x = -0.5 - (-1.0)
    assert abs(out["kl_k3"].item() - (math.exp(x) - x - 1.0)) < 1e-12, out["kl_k3"].item()
    wrong = math.exp(-x) - (-x) - 1.0
    assert abs(out["kl_k3"].item() - wrong) > 1e-3
    assert out["kl_k3"].item() >= 0.0


def check_clip_frac_strict_boundary(solve):
    # eps_low = eps_high = 0：rho 恰为 1 的 token 不算越界（严格不等号）
    lp = torch.tensor([[0.0, 0.05]], dtype=torch.float64)
    m = torch.ones(1, 2, dtype=torch.bool)
    out = solve(lp, torch.zeros_like(lp), lp.clone(), m,
                torch.zeros(1, dtype=torch.float64), 0.0, 0.0)
    assert abs(out["clip_frac"].item() - 0.5) < 1e-12, out["clip_frac"].item()


def check_asymmetric_and_mask(solve):
    # 非对称半径：rho=1.35 在 (0.2,0.5) 内、在 (0.2,0.2) 外；masked token 不计数
    lp = torch.tensor([[math.log(1.35), 0.0, 9.0]], dtype=torch.float64)
    m = torch.tensor([[True, True, False]], dtype=torch.bool)
    adv = torch.tensor([1.0], dtype=torch.float64)
    old = torch.zeros_like(lp)
    wide = solve(lp, old, lp.clone(), m, adv, 0.2, 0.5)
    tight = solve(lp, old, lp.clone(), m, adv, 0.2, 0.2)
    assert abs(wide["clip_frac"].item() - 0.0) < 1e-12
    assert abs(tight["clip_frac"].item() - 0.5) < 1e-12
    # 只有 2 个有效 token：loss = -(1.35·1·log1.35 + 1.0·1·0)/2
    expected = -(1.35 * math.log(1.35)) / 2.0
    assert abs(wide["loss"].item() - expected) < 1e-12, (wide["loss"].item(), expected)


def check_float32_input(solve):
    out = solve(LOGPS.to(torch.float32), LOGPS_OLD.to(torch.float32),
                REF_LOGPS.to(torch.float32), MASK, ADV.to(torch.float32), 0.2, 0.2)
    assert out["loss"].dtype == torch.float64 and out["kl_k3"].dtype == torch.float64
    assert abs(out["loss"].item() - EXAMPLE["loss"]) < 1e-6, out["loss"].item()


def check_errors(solve):
    lp = torch.zeros(1, 2, dtype=torch.float64)
    m = torch.ones(1, 2, dtype=torch.bool)
    adv = torch.zeros(1, dtype=torch.float64)

    def bad(fn, desc):
        try:
            fn()
        except ValueError:
            return
        raise AssertionError(f"应抛 ValueError: {desc}")

    bad(lambda: solve([[0.0, 0.0]], lp, lp, m, adv), "logp 非张量")
    bad(lambda: solve(lp, lp[0], lp, m, adv), "logp_old 非二维")
    bad(lambda: solve(lp, lp, lp[:, :1], m, adv), "ref 形状不符")
    bad(lambda: solve(lp, lp, lp, m.to(torch.int64), adv), "mask 非 bool")
    bad(lambda: solve(lp, lp, lp, m[:, :1], adv), "mask 形状不符")
    bad(lambda: solve(lp, lp, lp, torch.zeros(1, 2, dtype=torch.bool), adv), "无有效 token")
    bad(lambda: solve(lp, lp, lp, m, torch.zeros(2, dtype=torch.float64)), "adv 形状不符")
    bad(lambda: solve(lp, lp, lp, m, adv.to(torch.float32)), "adv dtype 不符")
    bad(lambda: solve(lp, lp, lp, m, adv, 1.0, 0.2), "eps_low=1")
    bad(lambda: solve(lp, lp, lp, m, adv, 0.2, -0.1), "eps_high<0")
    bad(lambda: solve(lp, lp, torch.full((1, 2), 21.0, dtype=torch.float64), m, adv), "log_r>20")
    bad(lambda: solve(torch.full((1, 2), float("nan"), dtype=torch.float64), lp, lp, m, adv), "含 NaN")
