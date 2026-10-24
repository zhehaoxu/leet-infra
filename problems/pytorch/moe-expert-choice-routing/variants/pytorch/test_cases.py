import torch


def _oracle(logits, k_c):
    """独立：手动 sort，避免依赖 topk 内部实现。"""
    probs = torch.softmax(logits.float(), dim=-1)
    T, E = probs.shape
    mask = torch.zeros(T, E, dtype=torch.bool)
    for e in range(E):
        col = [(float(probs[t, e]), t) for t in range(T)]
        col.sort(key=lambda x: (-x[0], x[1]))
        for _, t in col[:k_c]:
            mask[t, e] = True
    weights = probs * mask.float()
    out = torch.zeros_like(weights)
    for t in range(T):
        s = float(weights[t].sum())
        if s > 0:
            for e in range(E):
                out[t, e] = float(weights[t, e]) / s
    return mask, out


def check_example(solve):
    logits = torch.tensor([[2.0, 1, 0], [0, 2.0, 1], [1, 0, 2.0],
                           [3.0, 3.0, 0], [0, 0, 1.0], [-1.0, -1.0, -1.0]])
    m, w = solve(logits, 2)
    exp_m = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                          [1, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=torch.bool)
    assert (m == exp_m).all(), m.int().tolist()
    exp_w = torch.tensor([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0],
                          [0.5, 0.5, 0], [0, 0, 1.0], [0, 0, 0]])
    assert torch.allclose(w, exp_w, atol=1e-6), w.tolist()


def check_column_sum_equals_kc(solve):
    torch.manual_seed(0)
    for _ in range(15):
        T = int(torch.randint(4, 15, (1,)))
        E = int(torch.randint(2, 6, (1,)))
        k_c = int(torch.randint(1, T + 1, (1,)))
        logits = torch.randn(T, E)
        m, _ = solve(logits, k_c)
        assert (m.sum(dim=0) == k_c).all(), (T, E, k_c, m.sum(dim=0).tolist())


def check_unassigned_token_is_zero(solve):
    logits = torch.tensor([[2.0, 1, 0], [0, 2.0, 1], [1, 0, 2.0],
                           [3.0, 3.0, 0], [0, 0, 1.0], [-1.0, -1.0, -1.0]])
    _, w = solve(logits, 2)
    assert (w[5] == 0).all(), w[5].tolist()


def check_matches_oracle(solve):
    torch.manual_seed(1)
    for _ in range(30):
        T = int(torch.randint(5, 15, (1,)))
        E = int(torch.randint(3, 6, (1,)))
        k_c = int(torch.randint(1, T + 1, (1,)))
        logits = torch.randn(T, E)
        m1, w1 = solve(logits, k_c)
        m2, w2 = _oracle(logits, k_c)
        assert (m1 == m2).all(), (T, E, k_c)
        assert torch.allclose(w1, w2, atol=1e-6), (T, E, k_c, w1.tolist(), w2.tolist())


def check_renorm_rows_sum_to_one_when_assigned(solve):
    torch.manual_seed(2)
    logits = torch.randn(8, 4)
    _, w = solve(logits, 3)
    row_sum = w.sum(-1)
    # 大部分行有非零和，等于 1；若某行全零也合法
    assert bool(((row_sum < 1e-6) | ((row_sum - 1).abs() < 1e-5)).all())


check_example.example = True
