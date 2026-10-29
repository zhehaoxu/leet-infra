import torch


def _oracle(dense, block):
    d = dense.tolist()
    r, c = len(d), len(d[0])
    br, bc = block
    nz = [[v != 0.0 for v in row] for row in d]

    m = max(sum(row) for row in nz)
    ell_values, ell_indices = [], []
    for i in range(r):
        cols = [j for j in range(c) if nz[i][j]]
        vals = [d[i][j] for j in cols]
        pad = m - len(cols)
        ell_values.append(vals + [0.0] * pad)
        ell_indices.append(cols + [-1] * pad)

    nr, nc = r // br, c // bc
    bsr_values, bsr_indices, bsr_indptr = [], [], [0]
    for bi in range(nr):
        cnt = 0
        for bj in range(nc):
            blk = [d[bi * br + a][bj * bc + b] for a in range(br) for b in range(bc)]
            if any(v != 0.0 for v in blk):
                bsr_values.append(blk)
                bsr_indices.append(bj)
                cnt += 1
        bsr_indptr.append(bsr_indptr[-1] + cnt)

    nnz = sum(sum(row) for row in nz)
    nb = len(bsr_indices)
    nbytes = [
        nnz * 4 + nnz * 8 + (r + 1) * 8,
        r * m * 4 + r * m * 8,
        nb * br * bc * 4 + nb * 8 + (nr + 1) * 8,
    ]
    def t(x, dtype):
        return torch.tensor(x, dtype=dtype) if len(x) else torch.zeros(0, dtype=dtype)
    return (
        t(ell_values, torch.float32).reshape(r, m),
        t(ell_indices, torch.long).reshape(r, m),
        t(bsr_values, torch.float32).reshape(nb, br * bc),
        t(bsr_indices, torch.long),
        t(bsr_indptr, torch.long),
        torch.tensor(nbytes, dtype=torch.long),
    )


def _ell_to_dense(values, indices, c):
    r = values.shape[0]
    out = torch.zeros(r, c)
    for i in range(r):
        for t in range(values.shape[1]):
            j = int(indices[i, t])
            if j >= 0:
                out[i, j] = values[i, t]
    return out


def _bsr_to_dense(values, indices, indptr, block, r, c):
    br, bc = block
    out = torch.zeros(r, c)
    for bi in range(indptr.numel() - 1):
        for t in range(int(indptr[bi]), int(indptr[bi + 1])):
            bj = int(indices[t])
            blk = values[t]
            for a in range(br):
                for b in range(bc):
                    out[bi * br + a, bj * bc + b] = blk[a * bc + b]
    return out


D = torch.tensor(
    [[1.0, 0.0, 0.0, 2.0], [0.0, 0.0, 3.0, 0.0], [4.0, 5.0, 0.0, 0.0], [0.0, 0.0, 0.0, 6.0]]
)


def check_example(solve):
    ev, ei, bv, bi, bp, nb = solve(D, (2, 2))
    assert torch.allclose(ev, torch.tensor([[1.0, 2.0], [3.0, 0.0], [4.0, 5.0], [6.0, 0.0]]), atol=1e-6), ev.tolist()
    assert ei.tolist() == [[0, 3], [2, -1], [0, 1], [3, -1]], ei.tolist()
    assert torch.allclose(
        bv,
        torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 3.0, 0.0], [4.0, 5.0, 0.0, 0.0], [0.0, 0.0, 0.0, 6.0]]),
        atol=1e-6,
    ), bv.tolist()
    assert bi.tolist() == [0, 1, 0, 1], bi.tolist()
    assert bp.tolist() == [0, 2, 4], bp.tolist()
    assert nb.tolist() == [112, 96, 120], nb.tolist()


check_example.example = True


def check_matches_oracle(solve):
    torch.manual_seed(61)
    for r, c, blk in [(4, 4, (2, 2)), (6, 8, (2, 4)), (4, 6, (1, 3)), (8, 8, (4, 4))]:
        d = torch.randn(r, c) * (torch.rand(r, c) > 0.55)
        got = solve(d, blk)
        exp = _oracle(d, blk)
        names = ["ell_values", "ell_indices", "bsr_values", "bsr_indices", "bsr_indptr", "nbytes"]
        for name, g, e in zip(names, got, exp):
            if name in ("ell_values", "bsr_values"):
                assert g.shape == e.shape and torch.allclose(g, e, atol=1e-6), f"{name} {r}x{c} {blk}"
            else:
                assert g.tolist() == e.tolist(), f"{name} {r}x{c} {blk}: {g.tolist()} vs {e.tolist()}"


def check_roundtrip_ell(solve):
    torch.manual_seed(62)
    for r, c in [(5, 7), (3, 3), (4, 9)]:
        d = torch.randn(r, c) * (torch.rand(r, c) > 0.5)
        ev, ei, *_ = solve(d, (1, 1))
        assert torch.allclose(_ell_to_dense(ev, ei, c), d, atol=1e-6), f"{r}x{c}"


def check_roundtrip_bsr(solve):
    torch.manual_seed(63)
    for r, c, blk in [(4, 6, (2, 3)), (8, 8, (4, 4)), (6, 4, (3, 2))]:
        d = torch.randn(r, c) * (torch.rand(r, c) > 0.6)
        _, _, bv, bi, bp, _ = solve(d, blk)
        assert torch.allclose(_bsr_to_dense(bv, bi, bp, blk, r, c), d, atol=1e-6), f"{r}x{c} {blk}"


def check_ell_padding(solve):
    d = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 3.0, 4.0]])
    ev, ei, *_ = solve(d, (1, 1))
    assert ev.shape == (3, 3), ev.shape
    assert ei.tolist() == [[0, -1, -1], [-1, -1, -1], [0, 1, 2]], ei.tolist()
    assert ev.tolist() == [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 3.0, 4.0]], ev.tolist()


def check_all_zero_matrix(solve):
    d = torch.zeros(4, 4)
    ev, ei, bv, bi, bp, nb = solve(d, (2, 2))
    assert ev.shape == (4, 0) and ei.shape == (4, 0), (ev.shape, ei.shape)
    assert bv.shape[0] == 0 and bi.numel() == 0, (bv.shape, bi.tolist())
    assert bp.tolist() == [0, 0, 0], bp.tolist()
    assert nb.tolist() == [0 * 4 + 0 * 8 + 5 * 8, 0, 0 + 0 + 3 * 8], nb.tolist()


def check_dense_matrix_no_padding(solve):
    d = torch.randn(4, 4) + 5.0
    ev, ei, bv, bi, bp, nb = solve(d, (2, 2))
    assert ev.shape == (4, 4), ev.shape
    assert torch.all(ei >= 0), "全稠密时不应有补齐位"
    assert bp.tolist() == [0, 2, 4], bp.tolist()
    assert bi.tolist() == [0, 1, 0, 1], bi.tolist()


def check_block_one_one(solve):
    d = torch.tensor([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])
    _, _, bv, bi, bp, nb = solve(d, (1, 1))
    assert bv.shape == (3, 1), bv.shape
    assert bv.flatten().tolist() == [1.0, 2.0, 3.0], bv.tolist()
    assert bi.tolist() == [0, 2, 1], bi.tolist()
    assert bp.tolist() == [0, 2, 3], bp.tolist()
