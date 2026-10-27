import torch

E = torch.tensor([[1.0], [2.0]])                          # (2,1) r=2
A = torch.tensor([[3.0], [4.0]])                          # (2,1) d_in=1
B = torch.tensor([[5.0, 6.0]])                            # (1,2) d_out=1
ONES_E = torch.ones(2, 1)
ONES_A = torch.ones(2, 1)
ONES_B = torch.ones(1, 2)


def _element(p, g1, g2, b1, b2):
    ipt1 = abs(p * g1)
    ipt2 = abs(p * g2)
    avg = (1 - b1) * ipt1
    unc = (1 - b2) * abs(ipt1 - avg)
    avg = b1 * avg + (1 - b1) * ipt2
    unc = b2 * unc + (1 - b2) * abs(ipt2 - avg)
    return avg * unc


def _oracle(E, A, B, g1_E, g1_A, g1_B, g2_E, g2_A, g2_B, beta1, beta2, budget):
    """独立重算：逐标量两步 EMA，再按维度手工平均、排序找阈值（结构异于参考解）。"""
    r, d_in = A.shape
    d_out = B.shape[0]
    b1, b2 = float(beta1), float(beta2)
    sE = [_element(float(E[i, 0]), float(g1_E[i, 0]), float(g2_E[i, 0]), b1, b2)
          for i in range(r)]
    sA = [sum(_element(float(A[i, j]), float(g1_A[i, j]), float(g2_A[i, j]), b1, b2)
              for j in range(d_in)) / d_in for i in range(r)]
    sB = [sum(_element(float(B[w, i]), float(g1_B[w, i]), float(g2_B[w, i]), b1, b2)
              for w in range(d_out)) / d_out for i in range(r)]
    scores = [sE[i] + sA[i] + sB[i] for i in range(r)]
    masked = [[float(E[i, 0])] for i in range(r)]
    k = r - int(budget)
    if k >= 1:
        thresh = sorted(scores)[k - 1]
        for i in range(r):
            if scores[i] <= thresh:
                masked[i][0] = 0.0
    return (torch.tensor(scores, dtype=torch.float64).to(torch.float32),
            torch.tensor(masked, dtype=torch.float64).to(torch.float32))


def check_example(solve):
    scores, masked = solve(E, A, B, ONES_E, ONES_A, ONES_B, ONES_E, ONES_A, ONES_B, 0.5, 0.5, 1)
    assert torch.allclose(scores, torch.tensor([6.5625, 10.5]), atol=1e-5, rtol=1e-5), (
        f"got {scores.tolist()}")
    assert torch.allclose(masked, torch.tensor([[0.0], [2.0]]), atol=1e-5, rtol=1e-5), (
        f"got {masked.tolist()}")


check_example.example = True


def check_matches_oracle(solve):
    torch.manual_seed(23)
    r, d_in, d_out = 5, 6, 7
    e, a, b = torch.randn(r, 1), torch.randn(r, d_in), torch.randn(d_out, r)
    g1 = [torch.randn_like(t) for t in (e, a, b)]
    g2 = [torch.randn_like(t) for t in (e, a, b)]
    scores, masked = solve(e, a, b, g1[0], g1[1], g1[2], g2[0], g2[1], g2[2], 0.85, 0.9, 2)
    exp_s, exp_m = _oracle(e, a, b, g1[0], g1[1], g1[2], g2[0], g2[1], g2[2], 0.85, 0.9, 2)
    assert torch.allclose(scores, exp_s, atol=1e-5, rtol=1e-5), "scores 与独立重算不符"
    assert torch.allclose(masked, exp_m, atol=1e-5, rtol=1e-5), "masked_E 与独立重算不符"


def check_mask_applies_to_e_only(solve):
    """掩码只作用于 E：budget 是保留数，budget=1 应仅保留 1 个（删 r-1 个，连续分数无并列），
    且保留的是分数最高的 triplet，非零位保持原值。"""
    torch.manual_seed(5)
    r = 4
    e, a, b = torch.randn(r, 1), torch.randn(r, 3), torch.randn(6, r)
    g1 = [torch.randn_like(t) for t in (e, a, b)]
    g2 = [torch.randn_like(t) for t in (e, a, b)]
    scores, masked = solve(e, a, b, g1[0], g1[1], g1[2], g2[0], g2[1], g2[2], 0.8, 0.9, 1)
    zeros = int((masked[:, 0] == 0).sum())
    assert zeros == r - 1, f"budget=1 应保留 1 个（删 {r-1} 个），got 删 {zeros} 个"
    keep = (masked[:, 0] != 0).nonzero()[0, 0]
    assert torch.isclose(scores[keep], scores.max()), "保留的应是分数最高的 triplet"
    assert torch.isclose(masked[keep, 0], e[keep, 0]), "未被掩码的位置应保持 E 原值"


def check_budget_boundary(solve):
    torch.manual_seed(9)
    r = 3
    e, a, b = torch.randn(r, 1), torch.randn(r, 2), torch.randn(4, r)
    g1 = [torch.randn_like(t) for t in (e, a, b)]
    g2 = [torch.randn_like(t) for t in (e, a, b)]
    args = (e, a, b, g1[0], g1[1], g1[2], g2[0], g2[1], g2[2])
    _, m_full = solve(*args, 0.8, 0.8, r)
    assert torch.equal(m_full, e), "budget=r 时不做任何掩码，masked_E 应等于 E"
    _, m_one = solve(*args, 0.8, 0.8, r - 1)
    assert int((m_one[:, 0] == 0).sum()) == 1, "budget=r-1 应恰删 1 个"


def check_ties_delete_more(solve):
    """分数严格并列时，第 k 小阈值会把并列者一并置零（PEFT 原生行为，不去重）。"""
    r = 2
    e = torch.tensor([[1.0], [1.0]])
    a = torch.tensor([[2.0], [2.0]])
    b = torch.tensor([[3.0, 3.0]])
    g = torch.ones(1)
    args = (e, a, b, g.expand(r, 1), g.expand(r, 1), g.expand(1, r),
            g.expand(r, 1), g.expand(r, 1), g.expand(1, r))
    scores, masked = solve(*args, 0.5, 0.5, r - 1)
    assert torch.allclose(scores[0], scores[1]), "构造应为并列 score"
    assert int((masked[:, 0] == 0).sum()) == 2, "并列时 k=1 也会把两个都删掉"
