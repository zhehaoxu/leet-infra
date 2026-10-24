import torch


def _oracle(logits, k):
    """独立路径：python 排序 + renorm。"""
    probs = torch.softmax(logits.float(), dim=-1)
    T, E = probs.shape
    scores = torch.zeros(T, k)
    indices = torch.zeros(T, k, dtype=torch.long)
    for t in range(T):
        order = sorted(range(E), key=lambda e: (-float(probs[t, e]), e))
        picked = order[:k]
        s = sum(float(probs[t, e]) for e in picked)
        for j, e in enumerate(picked):
            indices[t, j] = e
            scores[t, j] = float(probs[t, e]) / s
    return scores, indices


def check_example(solve):
    logits = torch.tensor([[2.0, 1.0, 0.5, -1.0, -2.0, 0.0],
                           [0.0, 0.0, 3.0, 2.4, -1.0, -1.0],
                           [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    sc, idx = solve(logits, 2)
    exp_s = torch.tensor([[0.731059, 0.268941],
                          [0.645656, 0.354344],
                          [0.731059, 0.268941]])
    exp_i = torch.tensor([[0, 1], [2, 3], [5, 4]])
    assert torch.allclose(sc, exp_s, atol=1e-5), sc.tolist()
    assert (idx == exp_i).all(), idx.tolist()


def check_matches_oracle(solve):
    torch.manual_seed(0)
    for _ in range(30):
        T = int(torch.randint(2, 8, (1,)))
        E = int(torch.randint(3, 10, (1,)))
        k = int(torch.randint(1, E + 1, (1,)))
        logits = torch.randn(T, E)
        s1, i1 = solve(logits, k)
        s2, i2 = _oracle(logits, k)
        assert torch.allclose(s1, s2, atol=1e-5), (k, logits.tolist())
        assert (i1 == i2).all(), (k, logits.tolist())


def check_rows_sum_to_one(solve):
    torch.manual_seed(1)
    logits = torch.randn(5, 8)
    sc, _ = solve(logits, 3)
    assert torch.allclose(sc.sum(-1), torch.ones(5), atol=1e-6)


def check_k_equals_E_reduces_to_softmax(solve):
    logits = torch.tensor([[1.0, 2.0, 3.0, -1.0]])
    sc, idx = solve(logits, 4)
    # k=E 时 scores 就是 softmax，但按降序排列；重建到原序对拍
    exp = torch.softmax(logits, dim=-1)
    recon = torch.zeros_like(exp).scatter_(1, idx, sc)
    assert torch.allclose(recon, exp, atol=1e-6), recon.tolist()
    assert sorted(int(x) for x in idx[0]) == [0, 1, 2, 3]


def check_indices_are_valid_experts(solve):
    torch.manual_seed(2)
    logits = torch.randn(4, 6)
    _, idx = solve(logits, 2)
    assert (idx >= 0).all() and (idx < 6).all()
    # 每行内不重复
    for t in range(4):
        assert len(set(int(x) for x in idx[t])) == 2


check_example.example = True
