import torch

# 示例期望值（实测回填）：boundaries=[1,3,3,7]、x=[0,3,5,9]
EXAMPLE_BOUNDARIES = [1.0, 3.0, 3.0, 7.0]
EXAMPLE_X = [0.0, 3.0, 5.0, 9.0]
EXAMPLE_IDX_LEFT = [0, 1, 3, 4]
EXAMPLE_IDX_RIGHT = [0, 3, 3, 4]
EXAMPLE_COUNTS_LEFT = [1, 1, 0, 1, 1]
EXAMPLE_COUNTS_RIGHT = [1, 0, 0, 2, 1]


def check_example(solve):
    b = torch.tensor(EXAMPLE_BOUNDARIES)
    x = torch.tensor(EXAMPLE_X)
    idx, counts = solve(b, x)
    assert torch.equal(idx, torch.tensor(EXAMPLE_IDX_LEFT)), idx.tolist()
    assert torch.equal(counts, torch.tensor(EXAMPLE_COUNTS_LEFT, dtype=torch.int64)), \
        counts.tolist()
    assert counts.shape == (5,)
    idx_r, counts_r = solve(b, x, right=True)
    assert torch.equal(idx_r, torch.tensor(EXAMPLE_IDX_RIGHT)), idx_r.tolist()
    assert torch.equal(counts_r, torch.tensor(EXAMPLE_COUNTS_RIGHT, dtype=torch.int64)), \
        counts_r.tolist()
    assert idx.dtype == torch.int64


check_example.example = True


def check_matches_native(solve):
    """对拍 torch.searchsorted + torch.bincount（含加权、多维 x、int/float）。"""
    for seed in range(5):
        torch.manual_seed(20 + seed)
        b = torch.randn(7).sort().values
        x = torch.randn(40) * 3.0
        for right in (False, True):
            idx, counts = solve(b, x, right=right)
            exp_idx = torch.searchsorted(b, x, right=right)
            assert torch.equal(idx, exp_idx), (seed, right)
            exp_counts = torch.bincount(exp_idx, minlength=8)
            assert torch.equal(counts, exp_counts), (seed, right)
            assert counts.shape == (8,)
            # 加权
            w = torch.rand(40)
            _, wcounts = solve(b, x, right=right, weights=w)
            exp_w = torch.bincount(exp_idx, weights=w, minlength=8)
            assert torch.allclose(wcounts, exp_w, atol=1e-6), (seed, right)
            assert wcounts.dtype == torch.float32
    # 多维 x
    torch.manual_seed(31)
    b = torch.randn(6).sort().values
    x = torch.randn(5, 8) * 2.0
    for right in (False, True):
        idx, counts = solve(b, x, right=right)
        assert torch.equal(idx, torch.searchsorted(b, x, right=right))
        assert idx.shape == x.shape
        assert torch.equal(counts, torch.bincount(idx.reshape(-1), minlength=7))
    # int64 一组
    bi = torch.tensor([1, 3, 3, 7])
    xi = torch.tensor([0, 3, 5, 9, 3, 1])
    for right in (False, True):
        idx, counts = solve(bi, xi, right=right)
        assert torch.equal(idx, torch.searchsorted(bi, xi, right=right))
        assert torch.equal(counts, torch.bincount(idx, minlength=5))
    # float64 一组
    b64 = torch.randn(5, dtype=torch.float64).sort().values
    x64 = torch.randn(12, dtype=torch.float64)
    for right in (False, True):
        idx, counts = solve(b64, x64, right=right)
        assert torch.equal(idx, torch.searchsorted(b64, x64, right=right))
        assert counts.dtype == torch.int64


def _oracle_scan(b, x, right, w=None):
    """独立线性扫描 oracle：逐元素走 boundary，与比较求和的解法路径不同。"""
    bl = b.tolist()
    xl = x.tolist()
    wf = w.flatten().tolist() if w is not None else None
    flat = x.flatten().tolist()
    idx = []
    for v in flat:
        i = 0
        if right:
            while i < len(bl) and bl[i] <= v:
                i += 1
        else:
            while i < len(bl) and bl[i] < v:
                i += 1
        idx.append(i)
    counts = []
    for i in range(len(bl) + 1):
        if wf is None:
            counts.append(idx.count(i))
        else:
            counts.append(sum(wf[k] for k, t in enumerate(idx) if t == i))
    idx_t = torch.tensor(idx, dtype=torch.long).reshape(x.shape)
    dt = torch.float64 if (w is not None and w.dtype == torch.float64) else \
        (torch.float32 if wf is not None else torch.int64)
    return idx_t, torch.tensor(counts, dtype=dt)


def check_linear_scan_oracle(solve):
    torch.manual_seed(40)
    cases = [
        (torch.tensor([1.0, 3.0, 3.0, 7.0]), torch.randn(20) * 4.0, None),
        (torch.tensor([2.0, 2.0, 2.0, 5.0, 5.0]), torch.randn(15) * 3.0, None),
        (torch.randn(9).sort().values, torch.randn(30), torch.rand(30)),
        (torch.tensor([], dtype=torch.float32), torch.randn(8), None),
        (torch.tensor([1, 3, 3, 7]), torch.randint(-2, 10, (12,)), None),
        (torch.tensor([], dtype=torch.int64), torch.randint(0, 5, (6,)), None),
        (torch.tensor([-2.0, -2.0, 0.0, 4.0]), torch.randn(3, 4), torch.rand(3, 4)),
    ]
    for k, (b, x, w) in enumerate(cases):
        for right in (False, True):
            got_idx, got_counts = solve(b, x, right=right, weights=w)
            exp_idx, exp_counts = _oracle_scan(b, x, right, w)
            assert torch.equal(got_idx, exp_idx), (k, right, got_idx.tolist(), exp_idx.tolist())
            if w is None:
                assert torch.equal(got_counts, exp_counts), (k, right, got_counts.tolist())
            else:
                assert torch.allclose(got_counts, exp_counts, atol=1e-6), \
                    (k, right, got_counts.tolist(), exp_counts.tolist())


def check_duplicate_ties(solve):
    """重复 boundary 时 left/right 的具体下标差异（显式断言）。"""
    b = torch.tensor([2.0, 2.0, 2.0, 5.0])
    x = torch.tensor([1.0, 2.0, 3.0, 5.0, 6.0])
    idx, _ = solve(b, x, right=False)
    assert torch.equal(idx, torch.tensor([0, 0, 3, 3, 4])), idx.tolist()
    idx, _ = solve(b, x, right=True)
    assert torch.equal(idx, torch.tensor([0, 3, 3, 4, 4])), idx.tolist()
    # 计数：x=[2,2,5]（重复值落桶）
    x2 = torch.tensor([2.0, 2.0, 5.0])
    _, c = solve(b, x2, right=False)
    assert torch.equal(c, torch.tensor([2, 0, 0, 1, 0], dtype=torch.int64)), c.tolist()
    _, c = solve(b, x2, right=True)
    assert torch.equal(c, torch.tensor([0, 0, 0, 2, 1], dtype=torch.int64)), c.tolist()


def check_out_of_range_sides(solve):
    b = torch.tensor([1.0, 3.0, 7.0])
    x = torch.tensor([-5.0, 0.0, 100.0])
    idx, counts = solve(b, x, right=False)
    assert torch.equal(idx, torch.tensor([0, 0, 3])), idx.tolist()
    idx, counts = solve(b, x, right=True)
    assert torch.equal(idx, torch.tensor([0, 0, 3])), idx.tolist()
    assert torch.equal(counts, torch.tensor([2, 0, 0, 1], dtype=torch.int64))
    # 恰好等于端点 boundary：left / right 相差 1
    x2 = torch.tensor([1.0, 3.0, 7.0])
    idx, _ = solve(b, x2, right=False)
    assert torch.equal(idx, torch.tensor([0, 1, 2])), idx.tolist()
    idx, _ = solve(b, x2, right=True)
    assert torch.equal(idx, torch.tensor([1, 2, 3])), idx.tolist()


def check_empty_x(solve):
    b = torch.tensor([1.0, 3.0, 3.0, 7.0])
    x = torch.zeros(0)
    idx, counts = solve(b, x)
    assert idx.shape == (0,) and idx.dtype == torch.int64
    assert torch.equal(counts, torch.zeros(5, dtype=torch.int64))
    # 加权空 x：counts 全 0 且为浮点
    _, counts = solve(b, x, weights=torch.zeros(0))
    assert torch.equal(counts, torch.zeros(5, dtype=torch.float32))
    # 2-D 空 x
    idx, counts = solve(b, torch.zeros(0, 3))
    assert idx.shape == (0, 3)
    assert torch.equal(counts, torch.zeros(5, dtype=torch.int64))


def check_empty_boundaries(solve):
    """boundaries 为空：idx 全 0、counts 长度 1（全部元素落在唯一开区间桶）。"""
    b = torch.zeros(0)
    x = torch.tensor([1.0, 2.0])
    for right in (False, True):
        idx, counts = solve(b, x, right=right)
        assert torch.equal(idx, torch.zeros(2, dtype=torch.int64)), idx.tolist()
        assert counts.shape == (1,)
        assert torch.equal(counts, torch.tensor([2], dtype=torch.int64)), counts.tolist()
    # 加权：counts = 权重之和
    _, counts = solve(b, x, weights=torch.tensor([0.5, 1.5]))
    assert torch.equal(counts, torch.tensor([2.0], dtype=torch.float32))
    # x 也为空
    idx, counts = solve(b, torch.zeros(0))
    assert idx.shape == (0,)
    assert torch.equal(counts, torch.zeros(1, dtype=torch.int64))


def check_weighted_vs_unweighted(solve):
    b = torch.tensor([1.0, 3.0, 7.0])
    x = torch.tensor([0.0, 2.0, 2.0, 5.0, 9.0])
    w = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    idx, counts = solve(b, x)
    assert torch.equal(idx, torch.tensor([0, 1, 1, 2, 3], dtype=torch.int64))
    assert torch.equal(counts, torch.tensor([1, 2, 1, 1], dtype=torch.int64)), counts.tolist()
    assert counts.dtype == torch.int64
    _, wcounts = solve(b, x, weights=w)
    # 桶 1 收 2+3，桶 2 收 4，桶 3 收 5（M=3 -> counts 长度 4）
    assert torch.allclose(wcounts, torch.tensor([1.0, 5.0, 4.0, 5.0]), atol=1e-6), \
        wcounts.tolist()
    assert wcounts.dtype == torch.float32
    assert abs(wcounts.sum().item() - w.sum().item()) < 1e-6
    assert counts.sum().item() == x.numel()
    # float64 权重 -> float64 counts
    _, w64 = solve(b, x, weights=w.to(torch.float64))
    assert w64.dtype == torch.float64
