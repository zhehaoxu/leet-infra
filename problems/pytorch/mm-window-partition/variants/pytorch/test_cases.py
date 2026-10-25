import torch


def _oracle(x, w):
    B, H, Wd, C = x.shape
    nr, nc = H // w, Wd // w
    out = torch.zeros(B * nr * nc, w, w, C)
    for b in range(B):
        for br in range(nr):
            for bc in range(nc):
                wi = b * nr * nc + br * nc + bc
                for i in range(w):
                    for j in range(w):
                        out[wi, i, j] = x[b, br * w + i, bc * w + j]
    return out


def _merge(windows, B, H, Wd, w, C):
    nr, nc = H // w, Wd // w
    x = windows.view(B, nr, nc, w, w, C).permute(0, 1, 3, 2, 4, 5)
    return x.reshape(B, H, Wd, C)


def check_example(solve):
    x = torch.arange(32, dtype=torch.float32).reshape(1, 4, 4, 2)
    y = solve(x, 2)
    assert y.shape == (4, 2, 2, 2), y.shape
    exp0 = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[8.0, 9.0], [10.0, 11.0]]])
    assert torch.allclose(y[0], exp0, atol=1e-5), f"got {y[0].tolist()}"


check_example.example = True


def check_matches_oracle(solve):
    torch.manual_seed(51)
    for B, H, Wd, C, w in [(1, 6, 6, 3, 3), (2, 4, 8, 2, 2), (1, 4, 4, 5, 2)]:
        x = torch.randn(B, H, Wd, C)
        assert torch.allclose(solve(x, w), _oracle(x, w), atol=1e-5)


def check_shape(solve):
    x = torch.randn(2, 8, 6, 4)
    y = solve(x, 2)
    assert y.shape == (2 * (8 // 2) * (6 // 2), 2, 2, 4), y.shape


def check_roundtrip(solve):
    torch.manual_seed(52)
    B, H, Wd, C, w = 2, 6, 8, 3, 2
    x = torch.randn(B, H, Wd, C)
    y = solve(x, w)
    assert torch.allclose(_merge(y, B, H, Wd, w, C), x, atol=1e-6)


def check_bijection(solve):
    torch.manual_seed(53)
    x = torch.randn(1, 4, 4, 3)
    y = solve(x, 2)
    assert torch.equal(torch.sort(y.flatten()).values, torch.sort(x.flatten()).values)
