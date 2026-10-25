import torch


def _oracle(prob, target, smoothing, reduction):
    N = prob.shape[0]
    vals = []
    for i in range(N):
        inter = sum(float(prob[i, c]) * float(target[i, c]) for c in range(prob.shape[1]))
        denom = sum(float(prob[i, c]) for c in range(prob.shape[1])) + \
                sum(float(target[i, c]) for c in range(prob.shape[1]))
        vals.append(1.0 - (2 * inter + smoothing) / (denom + smoothing))
    if reduction == "none":
        return torch.tensor(vals, dtype=torch.float32)
    if reduction == "sum":
        return torch.tensor(sum(vals), dtype=torch.float32)
    return torch.tensor(sum(vals) / N, dtype=torch.float32)


def check_example(solve):
    prob = torch.tensor([[0.9, 0.1]])
    target = torch.tensor([[1.0, 0.0]])
    out = solve(prob, target, 1.0)
    assert abs(out.item() - (1 - 2.8 / 3.0)) < 1e-6, f"got {out.item()}"


check_example.example = True


def check_perfect_prediction_zero(solve):
    # 二值掩膜完全一致 -> inter=ones, denom=2*ones -> dice=1 -> loss=0
    prob = torch.tensor([[0.0, 1.0, 1.0], [1.0, 0.0, 0.0]])
    target = prob.clone()
    out = solve(prob, target, 1.0, "none")
    assert torch.allclose(out, torch.zeros(2), atol=1e-5), out.tolist()


def check_opposite_hard(solve):
    # 完全不重叠、无平滑 -> dice=0 -> loss=1
    prob = torch.tensor([[1.0, 0.0]])
    target = torch.tensor([[0.0, 1.0]])
    out = solve(prob, target, 0.0)
    assert abs(out.item() - 1.0) < 1e-6, out.item()


def check_matches_oracle(solve):
    torch.manual_seed(8)
    prob = torch.rand(6, 4)
    target = (torch.rand(6, 4) > 0.5).float()
    for s in (0.0, 1.0):
        for red in ("mean", "sum", "none"):
            y = solve(prob, target, s, red)
            assert torch.allclose(y, _oracle(prob, target, s, red), atol=1e-5), (s, red)


def check_invalid_reduction(solve):
    prob = torch.rand(2, 3)
    target = torch.ones(2, 3)
    try:
        solve(prob, target, 1.0, "bogus")
    except ValueError:
        return
    raise AssertionError("非法 reduction 应抛出 ValueError")
