import torch

ORDER = (0, 2, 4, 6, 1, 3, 5, 7)


def _encode(codes, zeros):
    """测试侧编码器：按 tile / 半字 / 数据位顺序把逻辑码写进 int32。"""
    K, N = codes.shape
    qw = torch.zeros(K // 16, N // 16, 16, 2, dtype=torch.int32)
    for k in range(K):
        for c in range(N):
            t, r = divmod(k, 16)
            g, cc = divmod(c, 16)
            h, j = divmod(cc, 8)
            qw[t, g, r, h] |= (codes[k, c].to(torch.int32) & 0xF) << (4 * ORDER[j])
    qz = torch.zeros(N // 16, 2, dtype=torch.int32)
    for c in range(N):
        g, cc = divmod(c, 16)
        h, j = divmod(cc, 8)
        qz[g, h] |= (zeros[c].to(torch.int32) & 0xF) << (4 * ORDER[j])
    return qw, qz


def _oracle(qw, qz, scales, perm, K, N):
    """独立 oracle：按「本地列 j 读槽位 ORDER[j]」逐元素解码，用嵌套 list 组装。"""
    W = [[0.0] * N for _ in range(K)]
    for k in range(K):
        t, r = divmod(k, 16)
        for g in range(N // 16):
            for h in range(2):
                word = int(qw[t, g, r, h])
                zword = int(qz[g, h])
                for j in range(8):
                    c = 16 * g + 8 * h + j
                    code = (word >> (4 * ORDER[j])) & 0xF
                    zero = (zword >> (4 * ORDER[j])) & 0xF
                    W[k][perm[c]] = (code - zero) * float(scales[perm[c]])
    return torch.tensor(W, dtype=torch.float32)


def _case(K, N, perm, scale_hi=1.0):
    codes = torch.randint(0, 16, (K, N), dtype=torch.int64)
    zeros = torch.randint(0, 16, (N,), dtype=torch.int64)
    scales = torch.linspace(0.25, scale_hi, N, dtype=torch.float32)
    qw, qz = _encode(codes, zeros)
    return codes, zeros, scales, qw, qz


def check_example(solve):
    K = N = 16
    codes = torch.tensor([[(3 * k + n) % 16 for n in range(N)] for k in range(K)])
    zeros = torch.tensor([n % 2 for n in range(N)])
    scales = torch.tensor([1.0] * 8 + [0.5] * 8)
    perm = list(range(N - 1, -1, -1))
    qw, qz = _encode(codes, zeros)
    W = solve(qw, qz, scales, perm, K, N)
    assert W.shape == (16, 16), f"形状应为 (16,16)，got {tuple(W.shape)}"
    exp = torch.tensor(
        [
            [14.0, 14.0, 12.0, 12.0, 10.0, 10.0, 8.0, 8.0],
            [1.0, 1.0, -1.0, 15.0, 13.0, 13.0, 11.0, 11.0],
            [4.0, 4.0, 2.0, 2.0, 0.0, 0.0, 14.0, 14.0],
            [7.0, 7.0, 5.0, 5.0, 3.0, 3.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(W[:4, :8], exp, atol=1e-5), f"got {W[:4, :8].tolist()}"


check_example.example = True


def check_identity_perm(solve):
    """perm 为恒等排列时，W 就是 (code - zero) * scale 的直读。"""
    K = N = 16
    torch.manual_seed(7)
    codes = torch.randint(0, 16, (K, N), dtype=torch.int64)
    zeros = torch.randint(0, 16, (N,), dtype=torch.int64)
    scales = torch.rand(N, dtype=torch.float32) + 0.1
    qw, qz = _encode(codes, zeros)
    W = solve(qw, qz, scales, list(range(N)), K, N)
    exp = (codes - zeros).to(torch.float32) * scales[None, :]
    assert torch.allclose(W, exp, atol=1e-5), "恒等 perm 下解码不直读"


def check_nibble_slot_order(solve):
    """列值沿 tile 内列变化：槽位序写错会得到列置换。"""
    K = N = 16
    codes = torch.tensor([[(7 * n) % 16 for n in range(N)] for _ in range(K)])
    zeros = torch.zeros(N, dtype=torch.int64)
    scales = torch.ones(N, dtype=torch.float32)
    qw, qz = _encode(codes, zeros)
    W = solve(qw, qz, scales, list(range(N)), K, N)
    exp = codes.to(torch.float32)
    assert torch.allclose(W, exp, atol=1e-5), f"列槽位解码错误，got {W[0].tolist()}"


def check_row_slot_order(solve):
    """行值沿 tile 内行变化：行 nibble 槽位写错会把行号读乱。"""
    K = N = 16
    codes = torch.tensor([[k % 16] * N for k in range(K)], dtype=torch.int64)
    zeros = torch.zeros(N, dtype=torch.int64)
    scales = torch.ones(N, dtype=torch.float32)
    qw, qz = _encode(codes, zeros)
    W = solve(qw, qz, scales, list(range(N)), K, N)
    exp = torch.arange(16, dtype=torch.float32)[:, None].repeat(1, N)
    assert torch.allclose(W, exp, atol=1e-5), f"行槽位解码错误，got {W[:, 0].tolist()}"


def check_zero_point_packing(solve):
    """zero-point 也走同一套 nibble 槽位，且按存储列索引。"""
    K = N = 16
    codes = torch.zeros(K, N, dtype=torch.int64)
    zeros = torch.arange(N, dtype=torch.int64) % 16
    scales = torch.ones(N, dtype=torch.float32)
    perm = list(range(N - 1, -1, -1))
    qw, qz = _encode(codes, zeros)
    W = solve(qw, qz, scales, perm, K, N)
    exp = torch.tensor([-(float(zeros[perm[n]])) for n in range(N)], dtype=torch.float32)
    assert torch.allclose(W[0], exp, atol=1e-5), f"got {W[0].tolist()}"
    assert torch.allclose(W[5], exp, atol=1e-5), "zero-point 应与行无关"


def check_tile_boundary(solve):
    """K、N 超过一个 tile 时，第二个 tile 不能复用第一个 tile 的索引。"""
    K = N = 32
    torch.manual_seed(11)
    codes, zeros, scales, qw, qz = _case(K, N, list(range(N)), 1.4)
    perm = list(range(N - 1, -1, -1))
    W = solve(qw, qz, scales, perm, K, N)
    exp = _oracle(qw, qz, scales, perm, K, N)
    assert torch.allclose(W, exp, atol=1e-5), "跨 tile 解码不符"
    assert not torch.allclose(W[:16], W[16:]), "第 2 个 tile 行不应等于第 1 个"


def check_perm_scatter(solve):
    """perm 是散射（存储列 c -> 逻辑列 perm[c]），不是反向 gather。"""
    K = N = 16
    torch.manual_seed(3)
    codes, zeros, scales, qw, qz = _case(K, N, list(range(N)), 1.1)
    perm = [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15]
    W = solve(qw, qz, scales, perm, K, N)
    for c, n in enumerate(perm):
        exp = (codes[:, c] - zeros[c]).to(torch.float32) * scales[n]
        assert torch.allclose(W[:, n], exp, atol=1e-5), f"存储列 {c} 应落到逻辑列 {n}"


def check_bad_shape(solve):
    K = N = 16
    codes, zeros, scales, qw, qz = _case(K, N, list(range(N)))
    for kk, nn in ((8, 16), (16, 8)):
        try:
            solve(qw, qz, scales, list(range(N)), kk, nn)
        except ValueError:
            continue
        raise AssertionError(f"K={kk}, N={nn} 应抛 ValueError")


def check_random_against_oracle(solve):
    """随机形状 / 排列 / scale / zero 上与独立 oracle 全等。"""
    g = torch.Generator().manual_seed(2026)
    for _ in range(30):
        K = 16 * int(torch.randint(1, 4, (1,), generator=g))
        N = 16 * int(torch.randint(1, 4, (1,), generator=g))
        perm = torch.randperm(N, generator=g).tolist()
        codes = torch.randint(0, 16, (K, N), generator=g, dtype=torch.int64)
        zeros = torch.randint(0, 16, (N,), generator=g, dtype=torch.int64)
        scales = torch.rand(N, generator=g, dtype=torch.float32) + 0.05
        qw, qz = _encode(codes, zeros)
        got = solve(qw, qz, scales, perm, K, N)
        want = _oracle(qw, qz, scales, perm, K, N)
        assert torch.allclose(got, want, atol=1e-5), f"K={K}, N={N} 与 oracle 不符"
        for c in range(N):
            assert torch.allclose(got[:, perm[c]], (codes[:, c] - zeros[c]).to(torch.float32) * scales[perm[c]], atol=1e-5)
