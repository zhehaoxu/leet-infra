import math
import torch
import torch.nn.functional as F


def _oracle(logits, target, gamma, alpha, reduction):
    N, C = logits.shape
    vals = []
    for i in range(N):
        row = [float(logits[i, c]) for c in range(C)]
        m = max(row)
        exps = [math.exp(x - m) for x in row]
        s = sum(exps)
        pt = exps[int(target[i])] / s
        v = -((1 - pt) ** gamma) * math.log(pt)
        if alpha is not None:
            v = alpha * v
        vals.append(v)
    if reduction == "none":
        return torch.tensor(vals, dtype=torch.float32)
    if reduction == "sum":
        return torch.tensor(sum(vals), dtype=torch.float32)
    return torch.tensor(sum(vals) / N, dtype=torch.float32)


def check_example(solve):
    logits = torch.tensor([[2.0, 0.5], [0.1, 1.0]])
    target = torch.tensor([0, 1])
    out = solve(logits, target, 2.0)
    assert abs(out.item() - 0.017603164538741112) < 1e-6, f"got {out.item()}"


check_example.example = True


def check_gamma_zero_is_cross_entropy(solve):
    torch.manual_seed(3)
    logits = torch.randn(20, 5)
    target = torch.randint(0, 5, (20,))
    ce = F.cross_entropy(logits, target, reduction="mean")
    assert torch.allclose(solve(logits, target, 0.0), ce, atol=1e-5)
    assert torch.allclose(solve(logits, target, 0.0, None, "none"),
                          F.cross_entropy(logits, target, reduction="none"), atol=1e-5)


def check_matches_oracle(solve):
    torch.manual_seed(4)
    logits = torch.randn(8, 4)
    target = torch.randint(0, 4, (8,))
    for g in (0.5, 2.0):
        for a in (None, 0.75):
            y = solve(logits, target, g, a)
            assert torch.allclose(y, _oracle(logits, target, g, a, "mean"), atol=1e-5), (g, a)


def check_alpha_scaling(solve):
    torch.manual_seed(5)
    logits = torch.randn(6, 3)
    target = torch.randint(0, 3, (6,))
    base = solve(logits, target, 2.0, None)
    assert torch.allclose(solve(logits, target, 2.0, 0.25), base * 0.25, atol=1e-6)


def check_invalid_reduction(solve):
    logits = torch.randn(3, 2)
    target = torch.tensor([0, 1, 0])
    try:
        solve(logits, target, 2.0, None, "bogus")
    except ValueError:
        return
    raise AssertionError("非法 reduction 应抛出 ValueError")
