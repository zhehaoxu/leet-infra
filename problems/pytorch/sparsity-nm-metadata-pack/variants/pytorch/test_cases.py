import torch


def _oracle(weight):
    vals, metas = [], []
    for row in weight.tolist():
        rv, nibs = [], []
        for start in range(0, len(row), 4):
            grp = row[start : start + 4]
            pos = [i for i in range(4) if grp[i] != 0.0]
            assert len(pos) == 2, f"每组必须恰好 2 个非零，得到 {pos}"
            rv += [grp[pos[0]], grp[pos[1]]]
            nibs.append(pos[0] + ((pos[1] - 1) << 2))
        p = ((len(nibs) + 7) // 8) * 8
        nibs += [0] * (p - len(nibs))
        metas.append([nibs[2 * j] | (nibs[2 * j + 1] << 4) for j in range(p // 2)])
        vals.append(rv)
    return torch.tensor(vals, dtype=torch.float32), torch.tensor(metas, dtype=torch.uint8)


def _rand_24(r, k, seed):
    g = torch.Generator().manual_seed(seed)
    w = torch.zeros(r, k)
    for i in range(r):
        for j in range(k // 4):
            pick = torch.randperm(4, generator=g)[:2].sort().values
            v = torch.randn(2, generator=g) + 2.0
            v = torch.where(v == 0, torch.ones(2), v)
            w[i, j * 4 + int(pick[0])] = v[0]
            w[i, j * 4 + int(pick[1])] = v[1]
    return w


def _decode(values, meta, k):
    out = []
    rows = meta.tolist()
    for i in range(values.shape[0]):
        row = [0.0] * k
        for j in range(k // 4):
            byte = rows[i][j // 2]
            nib = (byte & 0xF) if j % 2 == 0 else (byte >> 4)
            i0 = nib & 0x3
            i1 = ((nib >> 2) & 0x3) + 1
            row[j * 4 + i0] = float(values[i, 2 * j])
            row[j * 4 + i1] = float(values[i, 2 * j + 1])
        out.append(row)
    return torch.tensor(out)


W = torch.tensor([[0.0, 3.0, -2.0, 0.0, 5.0, 0.0, 0.0, 7.0]])


def check_example(solve):
    values, meta = solve(W)
    assert torch.allclose(values, torch.tensor([[3.0, -2.0, 5.0, 7.0]]), atol=1e-6), values.tolist()
    assert meta.dtype == torch.uint8 and meta.shape == (1, 4), (meta.dtype, meta.shape)
    assert meta.flatten().tolist() == [133, 0, 0, 0], meta.tolist()

    w12 = torch.tensor([[1.0, 0.0, 0.0, 2.0, 0.0, 3.0, 4.0, 0.0, 5.0, 6.0, 0.0, 0.0]])
    values, meta = solve(w12)
    assert torch.allclose(values, torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]), atol=1e-6)
    assert meta.flatten().tolist() == [88, 0, 0, 0], meta.tolist()


check_example.example = True


def check_matches_oracle(solve):
    for idx, (r, k) in enumerate([(2, 8), (3, 12), (2, 32), (1, 4), (2, 64)]):
        w = _rand_24(r, k, 100 + idx)
        values, meta = solve(w)
        ev, em = _oracle(w)
        assert torch.allclose(values, ev, atol=1e-6), f"{r}x{k} values"
        assert torch.equal(meta, em), f"{r}x{k} meta: {meta.tolist()} vs {em.tolist()}"


def check_roundtrip_decode(solve):
    for idx, (r, k) in enumerate([(2, 8), (3, 12), (4, 20), (2, 64)]):
        w = _rand_24(r, k, 200 + idx)
        values, meta = solve(w)
        assert torch.allclose(_decode(values, meta, k), w, atol=1e-6), f"{r}x{k} 解码失败"


def check_padding_zeros(solve):
    w = _rand_24(2, 12, 300)  # 3 组 -> 补到 8 nibble = 4 字节
    values, meta = solve(w)
    assert values.shape == (2, 6), values.shape
    assert meta.shape == (2, 4), meta.shape
    for row in meta.tolist():
        nibs = []
        for byte in row:
            nibs += [byte & 0xF, byte >> 4]
        assert nibs[3:] == [0] * 5, f"第 3 组之后的补位 nibble 必须为 0，得到 {nibs}"


def check_padding_byte_alignment(solve):
    for k, expect_bytes in [(4, 4), (8, 4), (32, 4), (36, 8), (64, 8)]:
        w = _rand_24(1, k, 400 + k)
        _, meta = solve(w)
        assert meta.shape == (1, expect_bytes), f"K={k} 期望 {expect_bytes} 字节，得到 {meta.shape}"


def check_values_follow_ascending_index(solve):
    # 组内 i0 必须小于 i1（升序存放）
    w = torch.tensor([[0.0, 0.0, 5.0, 9.0, 1.0, 2.0, 0.0, 0.0]])
    values, _ = solve(w)
    assert torch.allclose(values, torch.tensor([[5.0, 9.0, 1.0, 2.0]]), atol=1e-6), values.tolist()


def check_reverse_index_encoding(solve):
    # i0=2, i1=3（最靠后的组合）: nib = 2 + (2 << 2) = 10
    w = torch.tensor([[0.0, 0.0, 4.0, 6.0]])
    _, meta = solve(w)
    assert meta.flatten().tolist() == [10, 0, 0, 0], meta.tolist()
