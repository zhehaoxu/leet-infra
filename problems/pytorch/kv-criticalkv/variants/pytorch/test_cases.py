import torch

BASE = torch.tensor([[0.9, 0.2, 0.7, 0.1, 0.5, 0.3],
                     [0.4, 0.8, 0.1, 0.6, 0.2, 0.9]])
WOV = torch.tensor([[1.0, 3.0, 0.5, 2.0, 4.0, 0.2],
                    [2.0, 0.1, 5.0, 0.5, 1.0, 0.3]])
CR, FSR, EPS = 0.5, 0.5, 1e-4
FMAX = torch.finfo(torch.float32).max


def _oracle(scores, wov_norm, cr, fsr, eps):
    """独立向量化 oracle：stable sort + scatter（与 solution 的显式 Python 循环路径不同）。"""
    H, L = scores.shape
    n_kept = max(1, int(L * (1 - cr)))
    b1 = int((1 - cr) * L * fsr)
    if fsr > 0:
        b1 = max(1, b1)
    b1 = min(b1, L)
    final = (scores + eps) * wov_norm
    if b1 > 0:
        cand = torch.sort(-scores, dim=-1, stable=True).indices[:, :b1]
        final = final.scatter(-1, cand, FMAX)
    top = torch.sort(-final, dim=-1, stable=True).indices[:, :n_kept]
    kept = torch.sort(top, dim=-1).values
    return kept, final


def check_example(solve):
    kept, final = solve(BASE, WOV, CR, FSR, EPS)
    exp_kept = torch.tensor([[0, 1, 4], [0, 2, 5]])
    exp_final = torch.tensor([
        [3.4028234663852886e+38, 0.6003000140190125, 0.3500500023365021,
         0.20020000636577606, 2.0004000663757324, 0.06002000346779823],
        [0.8001999855041504, 0.08001000434160233, 0.5005000233650208,
         0.3000500202178955, 0.20010000467300415, 3.4028234663852886e+38],
    ])
    assert torch.equal(kept, exp_kept), kept
    assert torch.allclose(final, exp_final, atol=1e-5, rtol=1e-5), final


check_example.example = True


def check_matches_oracle(solve):
    torch.manual_seed(7)
    for _ in range(40):
        H = int(torch.randint(1, 5, (1,)))
        L = int(torch.randint(1, 25, (1,)))
        cr = float(torch.rand(1)) * 0.9
        fsr = [0.0, 0.2, 0.5, 1.0][int(torch.randint(0, 4, (1,)))]
        scores = torch.rand(H, L)
        wov = torch.rand(H, L) * 4
        gk, gf = solve(scores, wov, cr, fsr, 1e-4)
        ek, ef = _oracle(scores, wov, cr, fsr, 1e-4)
        assert torch.equal(gk, ek), (H, L, cr, fsr, gk, ek)
        assert torch.allclose(gf, ef, atol=1e-5, rtol=1e-5), (gf, ef)


def check_first_stage_invariance(solve):
    # fsr=1.0 且 cr=0.5 时 b1 == n_kept：保留集恰为原始分 top-k，value 加权不生效
    kept, _ = solve(BASE, WOV, 0.5, 1.0, EPS)
    exp = torch.tensor([[0, 2, 4], [1, 3, 5]])
    assert torch.equal(kept, exp), kept


def check_zero_value_norm(solve):
    # wov_norm 全 0：非保护 token 最终分同为 0，剩余名额按最小位置填充
    scores = torch.tensor([[0.2, 0.9, 0.1, 0.5, 0.3, 0.7]])
    kept, final = solve(scores, torch.zeros(1, 6), 0.5, 0.5, 1e-4)
    assert torch.equal(kept, torch.tensor([[0, 1, 2]])), kept
    assert torch.allclose(final, torch.tensor([[0.0, FMAX, 0.0, 0.0, 0.0, 0.0]]),
                          atol=1e-5, rtol=1e-5), final


def check_ratio_zero_keeps_all(solve):
    # cr=0 → n_kept=L：全部保留；final 仍含阶段1 scatter 的 max，不能拿未 scatter 的分比
    kept, final = solve(BASE, WOV, 0.0, 0.5, EPS)
    ek, ef = _oracle(BASE, WOV, 0.0, 0.5, EPS)
    assert kept.shape == (2, 6), kept.shape
    assert torch.equal(kept, torch.tensor([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]])), kept
    assert torch.allclose(final, ef, atol=1e-5, rtol=1e-5), final
    assert torch.equal(kept, ek), (kept, ek)


def check_stage1_subset(solve):
    # 不变量：阶段 1 命中位置必属最终保留集
    torch.manual_seed(11)
    scores = torch.rand(3, 17)
    wov = torch.rand(3, 17)
    cr, fsr = 0.45, 0.3
    kept, _ = solve(scores, wov, cr, fsr, 1e-4)
    b1 = max(1, int(17 * (1 - cr) * fsr))
    for h in range(3):
        stage1 = set(torch.sort(-scores[h], stable=True).indices[:b1].tolist())
        assert stage1.issubset(set(kept[h].tolist())), (h, stage1, kept[h])
