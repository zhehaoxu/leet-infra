import torch


def _oracle(logits, k):
    """独立路径：one_hot + 显式 python 累加。"""
    probs = torch.softmax(logits.float(), dim=-1)
    T, E = probs.shape
    _, sel = torch.topk(probs, k, dim=-1)
    counts = [0] * E
    for t in range(T):
        for j in range(k):
            counts[int(sel[t, j])] += 1
    f = [c / T for c in counts]
    P = [float(probs[:, e].mean()) for e in range(E)]
    s = sum(f[e] * P[e] for e in range(E))
    return torch.tensor(float(E * s))


def check_example(solve):
    logits = torch.tensor([[2.0, 0.0, 0.0, 0.0],
                           [0.0, 2.0, 0.0, 0.0],
                           [0.0, 0.0, 2.0, 0.0],
                           [0.0, 0.0, 0.0, 2.0],
                           [1.0, 1.0, 0.0, 0.0],
                           [0.0, 1.0, 1.0, 0.0]])
    loss = solve(logits, 1)
    assert abs(float(loss) - 1.025673) < 1e-4, float(loss)


def check_perfectly_balanced_lower_bound(solve):
    # k=1 完美均衡（每专家各分到 1 个、每行尖峰）→ aux = 1
    logits = torch.tensor([[3.0, 0.0, 0.0, 0.0],
                           [0.0, 3.0, 0.0, 0.0],
                           [0.0, 0.0, 3.0, 0.0],
                           [0.0, 0.0, 0.0, 3.0]])
    loss = solve(logits, 1)
    assert abs(float(loss) - 1.0) < 1e-4, float(loss)


def check_highly_unbalanced(solve):
    logits = torch.tensor([[3.0, 0.0, 0.0, 0.0],
                           [3.0, 0.0, 0.0, 0.0],
                           [3.0, 0.0, 0.0, 0.0],
                           [3.0, 0.0, 0.0, 0.0]])
    loss = solve(logits, 1)
    assert abs(float(loss) - 3.480194) < 1e-3, float(loss)


def check_matches_oracle(solve):
    torch.manual_seed(0)
    for _ in range(30):
        T = int(torch.randint(3, 15, (1,)))
        E = int(torch.randint(3, 8, (1,)))
        k = int(torch.randint(1, E + 1, (1,)))
        logits = torch.randn(T, E)
        got = float(solve(logits, k))
        exp = float(_oracle(logits, k))
        assert abs(got - exp) < 1e-4, (T, E, k, got, exp)


def check_k_greater_than_one_scales(solve):
    # k=E 时 f_e=1 (每个 token 选所有专家，count_e = T 每专家) → 归一化后 sum f_e=E
    # aux = E * Σ 1 * P_e = E * 1 = E
    torch.manual_seed(7)
    E = 6
    logits = torch.randn(4, E)
    loss = solve(logits, E)
    assert abs(float(loss) - float(E)) < 1e-4, float(loss)


check_example.example = True
