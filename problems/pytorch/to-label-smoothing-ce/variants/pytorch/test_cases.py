import math
import torch


def _oracle(logits, target, alpha):
    """独立纯 python：手算稳定 log_softmax（减最大值 + logsumexp），再组 label smoothing。"""
    N, V = len(logits), len(logits[0])
    tot = 0.0
    for i in range(N):
        mx = max(logits[i])
        exps = [math.exp(logits[i][j] - mx) for j in range(V)]
        s = sum(exps)
        logp = [logits[i][j] - mx - math.log(s) for j in range(V)]
        nll = -logp[target[i]]
        smooth = -sum(logp) / V
        tot += (1 - alpha) * nll + alpha * smooth
    return tot / N


def check_example(solve):
    logits = torch.tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 1.0], [1.0, 1.0, 3.0]])
    target = torch.tensor([0, 1, 2])
    out = solve(logits, target, 0.1)
    assert abs(out.item() - 0.436532) < 1e-5, out.item()


def check_alpha0_equals_ce(solve):
    torch.manual_seed(0)
    logits = torch.randn(5, 4)
    target = torch.randint(0, 4, (5,))
    out = solve(logits, target, 0.0)
    ce = torch.nn.functional.cross_entropy(logits, target, reduction='mean')
    assert abs(out.item() - ce.item()) < 1e-6, (out.item(), ce.item())


def check_matches_oracle(solve):
    torch.manual_seed(6)
    for _ in range(50):
        N = int(torch.randint(1, 6, (1,)))
        V = int(torch.randint(2, 8, (1,)))
        logits = torch.randn(N, V) * 3
        target = torch.randint(0, V, (N,))
        alpha = float(torch.rand(1)) * 0.4
        got = float(solve(logits, target, alpha).item())
        exp = _oracle(logits.tolist(), target.tolist(), alpha)
        assert abs(got - exp) < 1e-4, (got, exp)


def check_large_logits_stable(solve):
    logits = torch.tensor([[1000.0, 1000.0, 1000.0]])
    out = solve(logits, torch.tensor([0]), 0.1)
    assert math.isfinite(out.item()), out.item()
    # 全相等 logit → logp 均 -ln3；NLL 与平滑项都 = ln3 → loss = ln3
    assert abs(out.item() - math.log(3)) < 1e-4, out.item()


check_example.example = True
