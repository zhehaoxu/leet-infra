import torch

D = torch.tensor([[1.0, 0.0, 0.0, 2.0], [0.0, 0.0, 3.0, 0.0], [4.0, 5.0, 0.0, 0.0]])


def _oracle(d):
    r, c = d.shape
    indptr = [0]
    indices = []
    values = []
    for i in range(r):
        for j in range(c):
            if d[i, j].item() != 0.0:
                indices.append(j)
                values.append(d[i, j].item())
        indptr.append(len(indices))
    return indptr, indices, values


def _reconstruct(indptr, indices, values, r, c):
    out = torch.zeros(r, c)
    for i in range(r):
        for t in range(int(indptr[i]), int(indptr[i + 1])):
            out[i, int(indices[t])] = float(values[t])
    return out


def check_example(solve):
    ip, ix, vv = solve(D)
    assert ip.tolist() == [0, 2, 3, 5], ip.tolist()
    assert ix.tolist() == [0, 3, 2, 0, 1], ix.tolist()
    assert vv.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0], vv.tolist()


check_example.example = True


def check_matches_oracle(solve):
    torch.manual_seed(13)
    for shape in [(4, 6), (7, 3), (5, 5)]:
        d = torch.randn(*shape)
        d[torch.rand_like(d) < 0.5] = 0.0
        ip, ix, vv = solve(d)
        oip, oix, ovv = _oracle(d)
        assert ip.tolist() == oip, (ip.tolist(), oip)
        assert ix.tolist() == oix
        assert all(abs(a - b) < 1e-6 for a, b in zip(vv.tolist(), ovv))


def check_lengths_and_roundtrip(solve):
    torch.manual_seed(4)
    d = torch.randn(6, 9)
    d[torch.rand_like(d) < 0.4] = 0.0
    r, c = d.shape
    nnz = int((d != 0).sum().item())
    ip, ix, vv = solve(d)
    assert ip.numel() == r + 1
    assert ix.numel() == nnz and vv.numel() == nnz
    assert int(ip[0].item()) == 0 and int(ip[-1].item()) == nnz
    assert torch.allclose(_reconstruct(ip, ix, vv, r, c), d, atol=1e-6)


def check_all_zero_degenerate(solve):
    z = torch.zeros(3, 4)
    ip, ix, vv = solve(z)
    assert ip.tolist() == [0, 0, 0, 0], ip.tolist()
    assert ix.numel() == 0 and vv.numel() == 0


def check_dtypes(solve):
    ip, ix, vv = solve(D)
    assert ip.dtype == torch.int64 and ix.dtype == torch.int64, (ip.dtype, ix.dtype)
    assert vv.dtype == torch.float32, vv.dtype
