import torch

# 示例：q_len=3, kv_len=5, kv_offset=2, causal=True（0/1 指示阵，1=可看）
EXAMPLE_PATTERN = [[1.0, 1.0, 1.0, 0.0, 0.0],
                   [1.0, 1.0, 1.0, 1.0, 0.0],
                   [1.0, 1.0, 1.0, 1.0, 1.0]]


def _oracle(q_len, kv_len, seq_lens=None, kv_lens=None, kv_offset=0,
            causal=True, window=None, dtype=torch.float32):
    """独立双重循环 oracle：逐 (b,i,j) 显式判定四条可见性规则，与解法的
    向量化广播路径完全不同。lens 长度 1 时按 batch 广播。"""
    lens_lists = [l for l in (seq_lens, kv_lens) if l is not None]
    b = max([len(l) for l in lens_lists] + [1])

    def get(lens, bi):
        if lens is None:
            return None
        return lens[bi % len(lens)]

    fmin = torch.finfo(dtype).min
    out = torch.empty(b, 1, q_len, kv_len, dtype=dtype)
    for bi in range(b):
        sl = get(seq_lens, bi)
        kl = get(kv_lens, bi)
        for i in range(q_len):
            for j in range(kv_len):
                vis = True
                if causal and not (j <= i + kv_offset):
                    vis = False
                if window is not None and not (i - (j - kv_offset) < window):
                    vis = False
                if kl is not None and not (j < kl):
                    vis = False
                if sl is not None and not (i < sl):
                    vis = False
                out[bi, 0, i, j] = 0.0 if vis else fmin
    return out


def _cmp(solve, **kwargs):
    dtype = kwargs.pop("dtype", torch.float32)
    got = solve(dtype=dtype, **kwargs)
    exp = _oracle(dtype=dtype, **kwargs)
    assert got.shape == exp.shape and got.dtype == dtype, (got.shape, got.dtype)
    # 掩码值只有两种（0.0 / finfo.min），可直接逐位相等比较
    assert torch.equal(got, exp), f"{kwargs}\ngot {got}\nexp {exp}"
    return got


def _indicator(mask):
    return (mask == 0.0).to(torch.float32)


def check_example(solve):
    mask = solve(3, 5, kv_offset=2, causal=True)  # 默认 dtype=float32
    assert mask.shape == (1, 1, 3, 5), mask.shape
    assert mask.dtype == torch.float32, mask.dtype
    pat = _indicator(mask)
    exp = torch.tensor(EXAMPLE_PATTERN).reshape(1, 1, 3, 5)
    assert torch.equal(pat, exp), pat.tolist()
    # 不可看处的具体值恰为 finfo.min（示例中存在被遮位置）
    fmin = torch.finfo(torch.float32).min
    assert torch.all(mask[~(mask == 0.0)] == fmin)


check_example.example = True


def check_oracle_all_combos(solve):
    """独立双重循环 oracle 全组合对拍：causal x window x lens 至少 6 组。"""
    combos = [
        dict(q_len=4, kv_len=4, causal=True),
        dict(q_len=4, kv_len=6, kv_offset=2, causal=True),
        dict(q_len=3, kv_len=5, causal=False),                       # 全可看
        dict(q_len=5, kv_len=5, causal=True, window=2),
        dict(q_len=5, kv_len=7, causal=False, window=3, kv_offset=1),
        dict(q_len=4, kv_len=5, causal=True, kv_lens=[3, 5]),
        dict(q_len=4, kv_len=5, causal=True, seq_lens=[2, 3], kv_lens=[4, 5],
             window=2, kv_offset=1),
        dict(q_len=4, kv_len=5, causal=True, seq_lens=[2], kv_lens=[3, 5]),  # seq 广播
        dict(q_len=6, kv_len=4, causal=False, seq_lens=[4, 1, 6], kv_lens=[2]),
        dict(q_len=3, kv_len=3, causal=True, kv_lens=[0, 3]),        # 整 batch 全遮
    ]
    for kw in combos:
        _cmp(solve, **kw)


def check_tril_decomposition(solve):
    """causal-only + kv_offset=0 + 无 lens：下三角结构，与 torch.tril 布尔模式一致。"""
    for q, k in ((4, 4), (3, 6), (5, 2)):
        mask = solve(q, k, causal=True)
        tri = torch.tril(torch.ones(q, k, dtype=torch.bool)).reshape(1, 1, q, k)
        assert torch.equal(_indicator(mask) == 1.0, tri), (q, k)
        assert torch.equal(mask == 0.0, tri)


def check_kv_offset_shift(solve):
    """kv_offset=3：对角带整体右移 3（显式断言每行的可看集合）。"""
    mask = solve(3, 6, kv_offset=3, causal=True)
    exp_rows = [[1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1]]
    pat = _indicator(mask)[0, 0].to(torch.int32).tolist()
    assert pat == exp_rows, pat
    # 与 kv_offset=0 的普通因果 mask 不同
    assert not torch.equal(mask, solve(3, 6, causal=True))


def check_window_band(solve):
    """window=1：只保留主对角带（结合 causal 后严格 j-i==kv_offset）。"""
    mask = solve(3, 4, causal=True, window=1)
    assert _indicator(mask)[0, 0].to(torch.int32).tolist() == \
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
    # kv_offset=1 时对角带在 j=i+1
    mask2 = solve(3, 5, causal=True, window=1, kv_offset=1)
    assert _indicator(mask2)[0, 0].to(torch.int32).tolist() == \
        [[0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0]]


def check_large_window_degenerates(solve):
    """window 足够大时退化为纯因果（逐位相等）。"""
    for kw in (dict(q_len=4, kv_len=6, kv_offset=2),
               dict(q_len=3, kv_len=5)):
        a = solve(**kw, causal=True)
        b = solve(**kw, causal=True, window=10 ** 9)
        assert torch.equal(a, b), kw


def check_full_lens_equivalence(solve):
    """kv_lens 全等 kv_len / seq_lens 全等 q_len 时与不给 lens 逐位相同。"""
    q, k = 4, 6
    base = solve(q, k, kv_offset=1, causal=True)
    assert torch.equal(base, solve(q, k, kv_offset=1, causal=True, kv_lens=[k]))
    assert torch.equal(base, solve(q, k, kv_offset=1, causal=True,
                                   seq_lens=[q], kv_lens=[k]))
    # 多 batch 满额 lens：每个 batch 与单 batch 结果一致
    two = solve(q, k, kv_offset=1, causal=True, seq_lens=[q, q], kv_lens=[k, k])
    assert two.shape == (2, 1, q, k), two.shape
    assert torch.equal(two[0], base[0]) and torch.equal(two[1], base[0])


def check_pad_rows_and_cols(solve):
    """pad 查询行整行 finfo.min；pad KV 列整列 finfo.min。"""
    fmin = torch.finfo(torch.float32).min
    mask = solve(4, 5, seq_lens=[2, 3], causal=True)
    for b, sl in ((0, 2), (1, 3)):
        for i in range(sl):
            assert mask[b, 0, i].max() == 0.0, (b, i)  # 有效行至少对角可见（0.0 为行内最大值）
        for i in range(sl, 4):
            assert torch.all(mask[b, 0, i] == fmin), (b, i)
    mask2 = solve(3, 5, kv_lens=[2, 5], causal=True)
    for j in range(2, 5):
        assert torch.all(mask2[0, 0, :, j] == fmin), j
    # batch0 的有效列内仍是因果模式
    tri = torch.tril(torch.ones(3, 2, dtype=torch.bool))
    assert torch.equal(mask2[0, 0, :, :2] == 0.0, tri)


def check_dtype_float64(solve):
    mask = solve(3, 4, kv_offset=1, dtype=torch.float64)
    assert mask.dtype == torch.float64
    fmin64 = torch.finfo(torch.float64).min
    assert torch.all(mask[~(mask == 0.0)] == fmin64)
    assert torch.any(mask == 0.0) and torch.any(mask == fmin64)
    exp = _oracle(3, 4, kv_offset=1, dtype=torch.float64)
    assert torch.equal(mask, exp)


def check_zero_dims(solve):
    m = solve(0, 5, causal=True)
    assert m.shape == (1, 1, 0, 5) and m.numel() == 0
    m2 = solve(3, 0, seq_lens=[2, 3])
    assert m2.shape == (2, 1, 3, 0) and m2.numel() == 0
    m3 = solve(0, 0, kv_lens=[0])
    assert m3.shape == (1, 1, 0, 0)


def check_error_cases(solve):
    def expect_value_error(fn, desc):
        try:
            fn()
        except ValueError:
            return
        raise AssertionError(f"应抛 ValueError: {desc}")

    expect_value_error(lambda: solve(3, 4, window=0), "window=0")
    expect_value_error(lambda: solve(3, 4, window=-3), "window<0")
    expect_value_error(lambda: solve(3, 4, dtype=torch.int32), "非法 dtype")
    expect_value_error(lambda: solve(-1, 4), "q_len<0")
    expect_value_error(lambda: solve(3, -2), "kv_len<0")
    expect_value_error(lambda: solve(3, 4, kv_offset=-1), "kv_offset<0")
    expect_value_error(lambda: solve(3, 4, seq_lens=[2, 3, 4], kv_lens=[2, 4]),
                       "lens 长度不匹配且均 >1")
    expect_value_error(lambda: solve(3, 4, kv_lens=[5]), "kv_lens 项 > kv_len")
    expect_value_error(lambda: solve(3, 4, seq_lens=[4]), "seq_lens 项 > q_len")
    expect_value_error(lambda: solve(3, 4, kv_lens=[-1]), "lens 负数")
