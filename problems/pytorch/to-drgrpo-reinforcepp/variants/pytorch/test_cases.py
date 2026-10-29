import math

import torch

REWARDS = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
LOGPS = torch.tensor([[[0.3, -0.3, 0.0], [0.05, 0.0, -0.05]]], dtype=torch.float64)
MASK = torch.ones(1, 2, 3, dtype=torch.bool)
EPS = 0.2
EXAMPLE = {
    "adv_drgrpo": [[0.5, -0.5]],
    "adv_reinforcepp": [[1.0, -1.0]],
    "loss_drgrpo": 0.005140191682918349,
    "loss_reinforcepp": 0.010280383365836698,
}


def _oracle(rewards, logp, old, mask, eps):
    b = len(rewards)
    g = len(rewards[0])
    t_len = len(logp[0][0])
    flat = [rewards[i][j] for i in range(b) for j in range(g)]
    mu = sum(flat) / len(flat)
    var = sum((v - mu) ** 2 for v in flat) / len(flat)
    sd = math.sqrt(var)
    adv_dr = [[rewards[i][j] - sum(rewards[i]) / g for j in range(g)] for i in range(b)]
    adv_rp = [[0.0 if sd <= 1e-12 else (rewards[i][j] - mu) / sd for j in range(g)] for i in range(b)]

    def surr(rho, a):
        c = min(max(rho, 1.0 - eps), 1.0 + eps)
        return min(rho * a, c * a)

    tot = 0.0
    for i in range(b):
        for j in range(g):
            for t in range(t_len):
                if not mask[i][j][t]:
                    continue
                rho = math.exp(logp[i][j][t] - old[i][j][t])
                tot += surr(rho, adv_dr[i][j])
    loss_dr = -tot / (b * g * t_len)
    seqs = []
    for i in range(b):
        for j in range(g):
            s = 0.0
            n = 0
            for t in range(t_len):
                if not mask[i][j][t]:
                    continue
                rho = math.exp(logp[i][j][t] - old[i][j][t])
                s += surr(rho, adv_rp[i][j])
                n += 1
            seqs.append(s / n)
    return adv_dr, adv_rp, loss_dr, -sum(seqs) / len(seqs)


def check_example(solve):
    out = solve(REWARDS, LOGPS, torch.zeros_like(LOGPS), MASK, EPS)
    assert set(out) == {"adv_drgrpo", "adv_reinforcepp", "loss_drgrpo", "loss_reinforcepp"}
    assert out["adv_drgrpo"].tolist() == EXAMPLE["adv_drgrpo"], out["adv_drgrpo"].tolist()
    assert out["adv_reinforcepp"].tolist() == EXAMPLE["adv_reinforcepp"], out["adv_reinforcepp"].tolist()
    assert abs(out["loss_drgrpo"].item() - EXAMPLE["loss_drgrpo"]) < 1e-12, out["loss_drgrpo"].item()
    assert abs(out["loss_reinforcepp"].item() - EXAMPLE["loss_reinforcepp"]) < 1e-12, out["loss_reinforcepp"].item()
    assert out["adv_drgrpo"].dtype == torch.float64 and out["loss_drgrpo"].dim() == 0


check_example.example = True


def check_oracle(solve):
    torch.manual_seed(37)
    for _ in range(5):
        b, g, t = 2, 3, 4
        r = torch.randn(b, g, dtype=torch.float64)
        lp = torch.randn(b, g, t, dtype=torch.float64) * 0.4
        old = torch.randn(b, g, t, dtype=torch.float64) * 0.4
        m = torch.rand(b, g, t) > 0.2
        m[:, :, 0] = True
        out = solve(r, lp, old, m, 0.15)
        adv_dr, adv_rp, l_dr, l_rp = _oracle(r.tolist(), lp.tolist(), old.tolist(), m.tolist(), 0.15)
        assert torch.allclose(out["adv_drgrpo"], torch.tensor(adv_dr, dtype=torch.float64), atol=1e-12)
        assert torch.allclose(out["adv_reinforcepp"], torch.tensor(adv_rp, dtype=torch.float64), atol=1e-12)
        assert abs(out["loss_drgrpo"].item() - l_dr) < 1e-12, (out["loss_drgrpo"].item(), l_dr)
        assert abs(out["loss_reinforcepp"].item() - l_rp) < 1e-12, (out["loss_reinforcepp"].item(), l_rp)


def check_advantage_invariants(solve):
    torch.manual_seed(41)
    r = torch.randn(3, 4, dtype=torch.float64)
    lp = torch.zeros(3, 4, 2, dtype=torch.float64)
    out = solve(r, lp, lp.clone(), torch.ones(3, 4, 2, dtype=torch.bool), 0.2)
    assert torch.allclose(out["adv_drgrpo"].sum(1), torch.zeros(3, dtype=torch.float64), atol=1e-12)
    rp = out["adv_reinforcepp"]
    assert abs(rp.mean().item()) < 1e-12
    assert abs(rp.std(unbiased=False).item() - 1.0) < 1e-12


def check_padding_scaling(solve):
    # 追加一列全 mask=False：Dr. GRPO 因分母含 T 而缩小，REINFORCE++ 不变
    lp2 = torch.cat([LOGPS, torch.full((1, 2, 1), 9.0, dtype=torch.float64)], dim=2)
    mask2 = torch.cat([MASK, torch.zeros(1, 2, 1, dtype=torch.bool)], dim=2)
    base = solve(REWARDS, LOGPS, torch.zeros_like(LOGPS), MASK, EPS)
    padded = solve(REWARDS, lp2, torch.zeros_like(lp2), mask2, EPS)
    ratio = padded["loss_drgrpo"].item() / base["loss_drgrpo"].item()
    assert abs(ratio - 3.0 / 4.0) < 1e-12, ratio
    assert abs(padded["loss_reinforcepp"].item() - base["loss_reinforcepp"].item()) < 1e-12


def check_zero_std(solve):
    r = torch.full((2, 3), 0.7, dtype=torch.float64)
    lp = torch.zeros(2, 3, 2, dtype=torch.float64)
    out = solve(r, lp, lp.clone(), torch.ones(2, 3, 2, dtype=torch.bool), 0.2)
    assert torch.allclose(out["adv_reinforcepp"], torch.zeros(2, 3, dtype=torch.float64), atol=1e-12)
    assert abs(out["loss_reinforcepp"].item()) < 1e-12
    assert abs(out["loss_drgrpo"].item()) < 1e-12


def check_clipping(solve):
    # 两个样本的 rho 都是 exp(0.3)=1.34986，eps=0.01：正优势被裁到 1.01，
    # 负优势按 min(rho·A, clip·A) 取更负的 rho·A（PPO 的自动单向裁剪）
    lp = torch.full((1, 2, 1), 0.3, dtype=torch.float64)
    out = solve(torch.tensor([[1.0, 0.0]], dtype=torch.float64), lp,
                torch.zeros_like(lp), torch.ones(1, 2, 1, dtype=torch.bool), 0.01)
    surr_pos = 1.01 * 0.5
    surr_neg = math.exp(0.3) * (-0.5)
    exp_dr = -(surr_pos + surr_neg) / 2.0
    assert abs(out["loss_drgrpo"].item() - exp_dr) < 1e-12, (out["loss_drgrpo"].item(), exp_dr)
    assert math.exp(0.3) * 0.5 > surr_pos, "正优势应被上界裁掉"


def check_errors(solve):
    r = torch.ones(1, 2, dtype=torch.float64)
    lp = torch.zeros(1, 2, 2, dtype=torch.float64)
    m = torch.ones(1, 2, 2, dtype=torch.bool)

    def bad(fn, desc):
        try:
            fn()
        except ValueError:
            return
        raise AssertionError(f"应抛 ValueError: {desc}")

    bad(lambda: solve([[1.0, 0.0]], lp, lp, m), "rewards 非张量")
    bad(lambda: solve(r, lp[0], lp[0], m[0]), "logp 非三维")
    bad(lambda: solve(r, lp, lp[:, :, :1], m), "logp_old 形状不符")
    bad(lambda: solve(torch.ones(1, 3, dtype=torch.float64), lp, lp, m), "rewards 形状不符")
    bad(lambda: solve(torch.ones(1, 1, dtype=torch.float64), torch.zeros(1, 1, 2, dtype=torch.float64), torch.zeros(1, 1, 2, dtype=torch.float64), torch.ones(1, 1, 2, dtype=torch.bool)), "G=1")
    bad(lambda: solve(r, lp, lp, m.to(torch.int64)), "mask 非 bool")
    bad(lambda: solve(r, lp, lp, torch.zeros(1, 2, 2, dtype=torch.bool)), "有空回答")
    bad(lambda: solve(r, lp, lp, m, 0.0), "eps=0")
    bad(lambda: solve(r, torch.full((1, 2, 2), float("inf"), dtype=torch.float64), lp, m), "含 Inf")
