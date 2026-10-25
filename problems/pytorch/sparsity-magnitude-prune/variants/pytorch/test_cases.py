import torch

W = torch.tensor([1.0, -5.0, 3.0, -2.0, 0.5, -9.0])
S = 0.5


def _oracle(w, s):
    n = w.numel()
    k = int(round(s * n))
    vals = w.flatten().tolist()
    order = sorted(range(n), key=lambda i: (abs(vals[i]), i))
    out = list(vals)
    for i in order[:k]:
        out[i] = 0.0
    return torch.tensor(out, dtype=torch.float32).view_as(w)


def check_example(solve):
    y = solve(W, S)
    exp = torch.tensor([0.0, -5.0, 3.0, 0.0, 0.0, -9.0])
    assert torch.allclose(y, exp, atol=1e-5, rtol=1e-5), f"got {y.tolist()}"


check_example.example = True


def check_matches_oracle(solve):
    torch.manual_seed(7)
    for shape in [(24,), (3, 8), (2, 2, 5)]:
        w = torch.randn(*shape)
        for s in [0.0, 0.25, 0.5, 0.75, 1.0]:
            assert torch.allclose(solve(w, s), _oracle(w, s), atol=1e-5, rtol=1e-5)


def check_zero_and_full_sparsity(solve):
    assert torch.equal(solve(W, 0.0), W), "sparsity=0 应原样"
    assert torch.equal(solve(W, 1.0), torch.zeros_like(W)), "sparsity=1 应全零"


def check_exactly_k_pruned(solve):
    torch.manual_seed(3)
    w = torch.randn(40)
    for s in [0.1, 0.3, 0.6]:
        k = int(round(s * 40))
        y = solve(w, s)
        assert (y == 0).sum().item() == k, (s, (y == 0).sum().item(), k)


def check_kept_values_unchanged(solve):
    y = solve(W, S)
    # 未剪位置的值与符号必须与原张量逐一致
    mask = y != 0
    assert torch.equal(y[mask], W[mask])
