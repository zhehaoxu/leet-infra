import torch


def _cdf(probs, u):
    c = 0.0
    for i, p in enumerate(probs):
        c += p
        if u < c:
            return i
    return len(probs) - 1


def _oracle(Pl, draft, eps, u_new):
    """独立纯 Python：typical acceptance 逐位判定，遇拒替换并停；全接受加 bonus。
    Pl 为 list[list[float]]。"""
    gamma = len(draft)
    out = []
    for t in range(gamma):
        x = int(draft[t])
        if Pl[t][x] >= eps:
            out.append(x)
        else:
            out.append(_cdf(Pl[t], u_new))
            return out
    out.append(_cdf(Pl[gamma], u_new))
    return out


def check_example(solve):
    P = torch.tensor([[0.9, 0.05, 0.05],
                      [0.1, 0.2, 0.7],
                      [0.3, 0.3, 0.4],
                      [0.2, 0.6, 0.2]])
    out = solve(P, torch.tensor([0, 0, 2]), 0.1, 0.4)
    assert torch.equal(out, torch.tensor([0, 0, 2, 1])), out


def check_reject_first(solve):
    P = torch.tensor([[0.9, 0.05, 0.05],
                      [0.1, 0.2, 0.7],
                      [0.3, 0.3, 0.4],
                      [0.2, 0.6, 0.2]])
    out = solve(P, torch.tensor([2, 0, 0]), 0.5, 0.92)
    assert torch.equal(out, torch.tensor([1])), out


def check_reject_mid(solve):
    P = torch.tensor([[0.9, 0.05, 0.05],
                      [0.1, 0.2, 0.7],
                      [0.3, 0.3, 0.4],
                      [0.2, 0.6, 0.2]])
    out = solve(P, torch.tensor([0, 2, 0]), 0.5, 0.4)
    assert torch.equal(out, torch.tensor([0, 2, 1])), out


def check_matches_oracle(solve):
    torch.manual_seed(5)
    for _ in range(60):
        V = int(torch.randint(2, 6, (1,)))
        gamma = int(torch.randint(1, 5, (1,)))
        rows = gamma + 1
        raw = torch.rand(rows, V) + 0.05
        P = raw / raw.sum(dim=1, keepdim=True)
        draft = [int(x) for x in torch.randint(0, V, (gamma,))]
        eps = float(0.02 + torch.rand(1) * 0.6)
        u_new = float(torch.rand(1) * 0.98)
        out = solve(P, torch.tensor(draft), eps, u_new)
        exp = torch.tensor(_oracle(P.tolist(), draft, eps, u_new), dtype=torch.long)
        assert torch.equal(out, exp), (P.tolist(), draft, eps, u_new, out, exp)


check_example.example = True
