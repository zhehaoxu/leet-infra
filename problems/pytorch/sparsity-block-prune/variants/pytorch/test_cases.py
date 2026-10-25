import torch

W = torch.tensor([[1.0, -5.0, 3.0, -2.0], [0.5, 9.0, -2.0, -8.0]])


def _oracle(w, s, b):
    r, c = w.shape
    nb = c // b
    total = r * nb
    k = int(round(s * total))
    norms = []
    for i in range(r):
        for j in range(nb):
            blk = w[i, j * b:(j + 1) * b]
            norms.append(blk.abs().sum().item())
    order = sorted(range(total), key=lambda t: (norms[t], t))
    prune = set(order[:k])
    out = torch.zeros_like(w)
    for i in range(r):
        for j in range(nb):
            if i * nb + j not in prune:
                out[i, j * b:(j + 1) * b] = w[i, j * b:(j + 1) * b]
    return out


def check_example(solve):
    y = solve(W, 0.5, 2)
    exp = torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.5, 9.0, -2.0, -8.0]])
    assert torch.allclose(y, exp, atol=1e-5, rtol=1e-5), f"got {y.tolist()}"


check_example.example = True


def check_matches_oracle(solve):
    torch.manual_seed(17)
    for shape in [(3, 8), (5, 12), (4, 4)]:
        w = torch.randn(*shape) * 4
        for s in [0.0, 0.25, 0.5]:
            for b in [2, 4]:
                if shape[1] % b == 0:
                    assert torch.allclose(solve(w, s, b), _oracle(w, s, b), atol=1e-5, rtol=1e-5)


def check_blocks_all_or_nothing(solve):
    torch.manual_seed(6)
    w = torch.randn(4, 12)
    y = solve(w, 0.5, 3)
    blocks = y.reshape(4, 4, 3)
    nz = (blocks != 0).sum(dim=2)  # 每块非零数
    assert torch.all((nz == 0) | (nz == 3)), "块必须整体保留或整体剪除"


def check_bounds(solve):
    assert torch.allclose(solve(W, 0.0, 2), W, atol=1e-6)
    assert torch.equal(solve(W, 1.0, 2), torch.zeros_like(W))


def check_kept_block_unchanged(solve):
    y = solve(W, 0.5, 2)
    # 保留的行1块0 应与原值逐一相同
    assert torch.equal(y[1, 0:2], W[1, 0:2])
