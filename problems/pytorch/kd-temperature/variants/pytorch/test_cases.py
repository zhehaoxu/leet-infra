import math

import torch


def _softmax(vec):
    m = max(vec)
    ex = [math.exp(v - m) for v in vec]
    s = sum(ex)
    return [e / s for e in ex]


def _kl_vec(p, q):
    return sum(pc * math.log(pc / qc) for pc, qc in zip(p, q) if pc > 0)


def _oracle(s, t, T, reduction):
    srows = s.tolist()
    trows = t.tolist()
    losses = []
    for srow, trow in zip(srows, trows):
        ps = _softmax([v / T for v in srow])
        pt = _softmax([v / T for v in trow])
        losses.append((T * T) * _kl_vec(pt, ps))
    if reduction == "none":
        return torch.tensor(losses, dtype=torch.float32)
    if reduction == "sum":
        return torch.tensor(sum(losses), dtype=torch.float32)
    return torch.tensor(sum(losses) / len(losses), dtype=torch.float32)


def check_example(solve):
    s = torch.tensor([[2.0, 1.0, 0.1]])
    t = torch.tensor([[1.0, 2.0, 0.5]])
    out = solve(s, t, 2.0)
    assert abs(out.item() - 0.39207178) < 1e-5, out.item()


check_example.example = True


def check_T1_equals_softmax_kl(solve):
    s = torch.tensor([[2.0, 1.0, 0.1], [0.3, 0.7, 1.5]])
    t = torch.tensor([[1.0, 2.0, 0.5], [1.1, 0.2, 0.4]])
    out = solve(s, t, 1.0)
    tot = 0.0
    for srow, trow in zip(s.tolist(), t.tolist()):
        tot += _kl_vec(_softmax(trow), _softmax(srow))
    assert abs(out.item() - tot / 2) < 1e-5, (out.item(), tot / 2)


def check_identical_zero(solve):
    s = torch.tensor([[0.4, 1.2, 2.0], [1.0, 1.0, 1.0]])
    out = solve(s, s.clone(), 2.0)
    assert abs(out.item()) < 1e-6, out.item()


def check_matches_oracle(solve):
    torch.manual_seed(5)
    s = torch.randn(6, 5)
    t = torch.randn(6, 5)
    for T in (1.0, 2.0, 4.0):
        for red in ("mean", "sum", "none"):
            y = solve(s, t, T, red)
            assert torch.allclose(y, _oracle(s, t, T, red), atol=1e-5), (T, red)


def check_invalid_reduction(solve):
    s = torch.randn(2, 3)
    t = torch.randn(2, 3)
    try:
        solve(s, t, 2.0, "bogus")
    except ValueError:
        return
    raise AssertionError("非法 reduction 应抛出 ValueError")
