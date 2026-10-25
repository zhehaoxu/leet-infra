import torch

W = torch.tensor([[1.0, -5.0, 3.0, -2.0, 0.5, -9.0, 2.0, 8.0]])


def _oracle(w):
    orig = w.shape
    g = w.reshape(-1, 4).tolist()
    out = []
    for grp in g:
        order = sorted(range(4), key=lambda i: (abs(grp[i]), -i), reverse=True)
        keep = set(order[:2])
        out.append([grp[i] if i in keep else 0.0 for i in range(4)])
    return torch.tensor(out, dtype=torch.float32).reshape(orig)


def check_example(solve):
    y = solve(W)
    exp = torch.tensor([[0.0, -5.0, 3.0, 0.0, 0.0, -9.0, 0.0, 8.0]])
    assert torch.allclose(y, exp, atol=1e-5, rtol=1e-5), f"got {y.tolist()}"


check_example.example = True


def check_matches_oracle(solve):
    torch.manual_seed(11)
    for shape in [(4,), (2, 8), (3, 4, 12)]:
        w = torch.randn(*shape) * 5
        assert torch.allclose(solve(w), _oracle(w), atol=1e-5, rtol=1e-5)


def check_each_group_has_two_nonzero(solve):
    torch.manual_seed(5)
    w = torch.randn(6, 16)
    y = solve(w).reshape(-1, 4)
    nz_per_group = (y != 0).sum(dim=1)
    assert torch.all(nz_per_group == 2), nz_per_group.tolist()


def check_overall_sparsity_half(solve):
    torch.manual_seed(5)
    w = torch.randn(5, 40)
    y = solve(w)
    frac_zero = (y == 0).float().mean().item()
    assert abs(frac_zero - 0.5) < 1e-6, frac_zero


def check_groups_independent(solve):
    # 改动某一行不应影响另一行的剪枝结果
    w = torch.tensor([[1.0, -5.0, 3.0, -2.0], [0.5, -9.0, 2.0, 8.0]])
    base = solve(w)
    w2 = w.clone()
    w2[1] = w2[1] * 100.0
    alt = solve(w2)
    assert torch.allclose(base[0], alt[0], atol=1e-6), "行0 结果不应受行1 影响"
