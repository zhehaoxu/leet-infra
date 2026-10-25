import torch


def _oracle(img, p):
    B, C, H, W = img.shape
    nr, nc = H // p, W // p
    out = torch.zeros(B, nr * nc, C * p * p)
    for b in range(B):
        for br in range(nr):
            for bc in range(nc):
                tok = []
                for c in range(C):
                    for i in range(p):
                        for j in range(p):
                            tok.append(img[b, c, br * p + i, bc * p + j].item())
                out[b, br * nc + bc] = torch.tensor(tok)
    return out


def _unpatchify(tokens, p, C, H, W):
    B, N, D = tokens.shape
    nr, nc = H // p, W // p
    x = tokens.view(B, nr, nc, C, p, p).permute(0, 3, 1, 4, 2, 5)
    return x.reshape(B, C, H, W)


def check_example(solve):
    img = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    y = solve(img, 2)
    exp = torch.tensor([[[0.0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]]])
    assert torch.allclose(y, exp, atol=1e-5), f"got {y.tolist()}"


check_example.example = True


def check_matches_oracle(solve):
    torch.manual_seed(21)
    for B, C, H, W, p in [(2, 3, 6, 6, 3), (1, 2, 8, 4, 2), (3, 1, 4, 4, 2)]:
        img = torch.randn(B, C, H, W)
        assert torch.allclose(solve(img, p), _oracle(img, p), atol=1e-5)


def check_shape(solve):
    img = torch.randn(2, 3, 8, 6)
    y = solve(img, 2)
    assert y.shape == (2, (8 // 2) * (6 // 2), 3 * 2 * 2), y.shape


def check_roundtrip(solve):
    torch.manual_seed(22)
    B, C, H, W, p = 2, 3, 8, 6, 2
    img = torch.randn(B, C, H, W)
    y = solve(img, p)
    assert torch.allclose(_unpatchify(y, p, C, H, W), img, atol=1e-6)


def check_element_bijection(solve):
    # patchify 是置换：输出元素多重集等于输入
    torch.manual_seed(23)
    img = torch.randn(1, 3, 6, 6)
    y = solve(img, 3)
    assert torch.equal(torch.sort(y.flatten()).values, torch.sort(img.flatten()).values)
