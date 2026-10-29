import torch


def _naive(q_nope, q_pe, c_kv, k_pe, W_UK, W_UV, scale):
    """独立 oracle：不用权重吸收——先把每个 token 的 k / v 升维出来，再逐 token 加权求和。"""
    kn = torch.einsum("bsc,hkc->bhsk", c_kv.double(), W_UK.double())
    v = torch.einsum("bsc,hkc->bhsk", c_kv.double(), W_UV.double())
    scores = torch.einsum("bhk,bhsk->bhs", q_nope.double(), kn)
    scores = scores + torch.einsum("bhr,bsr->bhs", q_pe.double(), k_pe.double())
    probs = torch.softmax(scores * scale, dim=-1)
    return torch.einsum("bhs,bhsk->bhk", probs, v)


def check_example(solve):
    q_nope = torch.tensor([[[1.0, 0.5], [0.0, -1.0]]])
    q_pe = torch.tensor([[[0.5, 0.5], [-0.5, 0.5]]])
    c_kv = torch.tensor([[[1.0, 0.0], [0.5, 1.0], [-1.0, 0.5]]])
    k_pe = torch.tensor([[[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]]])
    W_UK = torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[0.5, 0.5], [0.0, 1.0]]])
    W_UV = torch.tensor([[[1.0, 0.5], [0.0, 1.0]], [[0.5, 0.0], [1.0, 1.0]]])

    got = solve(q_nope, q_pe, c_kv, k_pe, W_UK, W_UV, 0.5)
    assert got.shape == (1, 2, 2), got.shape
    assert got.dtype == torch.float32
    assert [[[round(float(v), 6) for v in head] for head in batch] for batch in got.tolist()] == [
        [[0.698159, 0.5], [0.146223, 0.632368]]
    ], got.tolist()


check_example.example = True


def check_absorbed_matches_naive(solve):
    torch.manual_seed(0)
    for batch, heads, head_dim, rope_dim, latent_dim, seq in (
        (2, 4, 8, 4, 16, 7),
        (1, 2, 3, 2, 5, 4),
        (3, 3, 4, 4, 4, 2),
    ):
        q_nope = torch.randn(batch, heads, head_dim, dtype=torch.float64)
        q_pe = torch.randn(batch, heads, rope_dim, dtype=torch.float64)
        c_kv = torch.randn(batch, seq, latent_dim, dtype=torch.float64)
        k_pe = torch.randn(batch, seq, rope_dim, dtype=torch.float64)
        W_UK = torch.randn(heads, head_dim, latent_dim, dtype=torch.float64)
        W_UV = torch.randn(heads, head_dim, latent_dim, dtype=torch.float64)

        for scale in (1.0, 0.3):
            got = solve(q_nope, q_pe, c_kv, k_pe, W_UK, W_UV, scale)
            want = _naive(q_nope, q_pe, c_kv, k_pe, W_UK, W_UV, scale)
            assert torch.allclose(got, want.to(torch.float32), atol=1e-5), (
                f"吸收形式与逐 token 升维不一致: B={batch}, H={heads}, d_c={latent_dim}, scale={scale}"
            )


def check_single_key_attention(solve):
    torch.manual_seed(1)
    heads, head_dim, rope_dim, latent_dim = 3, 4, 2, 6
    c_kv = torch.randn(2, 1, latent_dim, dtype=torch.float64)
    k_pe = torch.randn(2, 1, rope_dim, dtype=torch.float64)
    W_UK = torch.randn(heads, head_dim, latent_dim, dtype=torch.float64)
    W_UV = torch.randn(heads, head_dim, latent_dim, dtype=torch.float64)

    outs = []
    for _ in range(3):
        q_nope = torch.randn(2, heads, head_dim, dtype=torch.float64) * 5.0
        q_pe = torch.randn(2, heads, rope_dim, dtype=torch.float64) * 5.0
        outs.append(solve(q_nope, q_pe, c_kv, k_pe, W_UK, W_UV, 0.7))
    # 只有一个 key 时 softmax 恒为 1，注意力结果与查询无关
    for out in outs[1:]:
        assert torch.allclose(out, outs[0], atol=1e-6), "S=1 时输出不应依赖查询"

    want = torch.einsum("bsc,hkc->bhk", c_kv, W_UV)
    assert torch.allclose(outs[0], want.to(torch.float32), atol=1e-6), "S=1 时输出应等于 W_UV · c_kv"


def check_uniform_attention(solve):
    heads, head_dim, rope_dim, latent_dim, seq = 2, 3, 2, 5, 4
    torch.manual_seed(2)
    c_kv = torch.randn(2, seq, latent_dim, dtype=torch.float64)
    W_UK = torch.randn(heads, head_dim, latent_dim, dtype=torch.float64)
    W_UV = torch.randn(heads, head_dim, latent_dim, dtype=torch.float64)

    zeros_k = torch.zeros(2, heads, head_dim, dtype=torch.float64)
    zeros_r = torch.zeros(2, heads, rope_dim, dtype=torch.float64)
    zeros_pe = torch.zeros(2, seq, rope_dim, dtype=torch.float64)

    got = solve(zeros_k, zeros_r, c_kv, zeros_pe, W_UK, W_UV, 1.0)
    # 查询全零 => 打分全零 => 权重均匀 => 输出是 W_UV 作用在 c_kv 的均值上
    want = torch.einsum("bsc,hkc->bhk", c_kv.mean(dim=1, keepdim=True), W_UV)
    assert torch.allclose(got, want.to(torch.float32), atol=1e-6), "全零查询应给出均匀注意力"


def check_head_independence(solve):
    torch.manual_seed(3)
    batch, heads, head_dim, rope_dim, latent_dim, seq = 2, 3, 4, 2, 6, 5
    q_nope = torch.randn(batch, heads, head_dim, dtype=torch.float64)
    q_pe = torch.randn(batch, heads, rope_dim, dtype=torch.float64)
    c_kv = torch.randn(batch, seq, latent_dim, dtype=torch.float64)
    k_pe = torch.randn(batch, seq, rope_dim, dtype=torch.float64)
    W_UK = torch.randn(heads, head_dim, latent_dim, dtype=torch.float64)
    W_UV = torch.randn(heads, head_dim, latent_dim, dtype=torch.float64)

    tricky_UK = W_UK.clone()
    tricky_UK[0] = torch.randn(head_dim, latent_dim, dtype=torch.float64)
    tricky_UV = W_UV.clone()
    tricky_UV[0] = torch.randn(head_dim, latent_dim, dtype=torch.float64)
    # 只改第 0 个头时，其余头的输出必须完全不变（含 RoPE 分数）
    base = solve(q_nope, q_pe, c_kv, k_pe, W_UK, W_UV, 0.5)
    other = solve(q_nope, q_pe, c_kv, k_pe, tricky_UK, tricky_UV, 0.5)
    assert torch.allclose(base[:, 1:], other[:, 1:], atol=1e-7), "各头之间不应互相影响"
    assert not torch.allclose(base[:, 0], other[:, 0], atol=1e-4), "被改动的头输出应发生变化"


def check_rope_is_shared(solve):
    torch.manual_seed(4)
    batch, heads, head_dim, rope_dim, latent_dim, seq = 2, 2, 4, 3, 5, 6
    q_nope = torch.randn(batch, heads, head_dim, dtype=torch.float64)
    q_pe = torch.randn(batch, heads, rope_dim, dtype=torch.float64)
    c_kv = torch.randn(batch, seq, latent_dim, dtype=torch.float64)
    W_UK = torch.randn(heads, head_dim, latent_dim, dtype=torch.float64)
    W_UV = torch.randn(heads, head_dim, latent_dim, dtype=torch.float64)

    zero_pe = torch.zeros(batch, seq, rope_dim, dtype=torch.float64)
    with_rope = solve(q_nope, q_pe, c_kv, torch.randn(batch, seq, rope_dim, dtype=torch.float64), W_UK, W_UV, 0.5)
    no_rope = solve(q_nope, q_pe, c_kv, zero_pe, W_UK, W_UV, 0.5)
    assert not torch.allclose(with_rope, no_rope, atol=1e-4), "RoPE 键应影响打分"
    # 与逐 token 升维的 oracle 在两种情况下都对得上
    for k_pe in (zero_pe, torch.randn(batch, seq, rope_dim, dtype=torch.float64)):
        got = solve(q_nope, q_pe, c_kv, k_pe, W_UK, W_UV, 0.5)
        want = _naive(q_nope, q_pe, c_kv, k_pe, W_UK, W_UV, 0.5)
        assert torch.allclose(got, want.to(torch.float32), atol=1e-5)


def check_invalid(solve):
    q_nope = torch.zeros(1, 2, 3)
    q_pe = torch.zeros(1, 2, 2)
    c_kv = torch.zeros(1, 4, 5)
    k_pe = torch.zeros(1, 4, 2)
    W_UK = torch.zeros(2, 3, 5)
    W_UV = torch.zeros(2, 3, 5)

    for bad in (
        lambda: solve(q_nope[0], q_pe, c_kv, k_pe, W_UK, W_UV, 0.5),
        lambda: solve(q_nope, q_pe[0], c_kv, k_pe, W_UK, W_UV, 0.5),
        lambda: solve(q_nope, q_pe, c_kv[0], k_pe, W_UK, W_UV, 0.5),
        lambda: solve(q_nope, q_pe, c_kv, k_pe[0], W_UK, W_UV, 0.5),
        lambda: solve(q_nope, q_pe, c_kv, k_pe, W_UK[0], W_UV, 0.5),
        lambda: solve(q_nope, q_pe, c_kv, k_pe, W_UK, W_UV[0], 0.5),
        lambda: solve(q_nope, q_pe, c_kv, k_pe[:, :3], W_UK, W_UV, 0.5),
        lambda: solve(q_nope, q_pe, c_kv, k_pe[..., :1], W_UK, W_UV, 0.5),
        lambda: solve(q_nope, q_pe[:, :1], c_kv, k_pe, W_UK, W_UV, 0.5),
        lambda: solve(q_nope, q_pe, c_kv, k_pe, torch.zeros(3, 3, 5), W_UV, 0.5),
        lambda: solve(q_nope, q_pe, c_kv, k_pe, W_UK, torch.zeros(2, 3, 4), 0.5),
        lambda: solve(q_nope, q_pe, c_kv, k_pe, W_UK, W_UV, 0.0),
        lambda: solve(q_nope, q_pe, c_kv, k_pe, W_UK, W_UV, -0.5),
        lambda: solve(q_nope, q_pe, c_kv, k_pe, W_UK, W_UV, float("nan")),
        lambda: solve(q_nope, q_pe, c_kv, k_pe, W_UK, W_UV, "0.5"),
        lambda: solve([[0.0, 1.0]], q_pe, c_kv, k_pe, W_UK, W_UV, 0.5),
    ):
        try:
            bad()
            raise AssertionError("非法输入应抛 ValueError")
        except ValueError:
            pass
