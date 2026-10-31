import numpy as np
import torch

KEYS = {"sign", "page_max", "scores", "selected"}


def _np_dtype(dtype):
    return np.float32 if dtype == torch.float32 else np.float64


def _oracle(q, k, page_size, page_budget):
    """numpy 独立重算：逐通道按 sign 取端 → 页内 amax → 点积 → 去掉末页。"""
    qn = q.detach().numpy().astype(np.float64).reshape(-1, q.shape[2])   # (H, D)
    kn = np.transpose(k.detach().numpy().astype(np.float64), (1, 0, 2))  # (H, Lkv, D)
    num_heads, kv_len, head_dim = kn.shape
    num_pages = -(-kv_len // page_size)
    sentinel = float(np.finfo(_np_dtype(q.dtype)).min)

    sign = np.where(qn > 0, 1.0, -1.0)
    max_key = kn * sign[:, None, :]
    pad = num_pages * page_size - kv_len
    if pad:
        max_key = np.concatenate(
            [max_key, np.full((num_heads, pad, head_dim), sentinel)], axis=1)
    page_max = max_key.reshape(num_heads, num_pages, page_size, head_dim).max(axis=2)
    scores = np.einsum("hd,hpd->hp", qn * sign, page_max)[:, : num_pages - 1]

    keep = min(page_budget, num_pages) - 1
    selected = []
    for h in range(num_heads):
        order = sorted(range(num_pages - 1), key=lambda p: (-scores[h, p], p))
        selected.append(sorted(order[:keep] + [num_pages - 1]))
    return {
        "sign": torch.tensor(sign, dtype=q.dtype),
        "page_max": torch.tensor(page_max, dtype=q.dtype),
        "scores": torch.tensor(scores, dtype=q.dtype),
        "selected": torch.tensor(np.array(selected, dtype=np.int64).reshape(num_heads, keep + 1)),
    }


def _close(a, b, dtype=None):
    tol = 3e-5 if (dtype or a.dtype) == torch.float32 else 2e-9
    torch.testing.assert_close(a, b, rtol=tol, atol=tol)


def _check_structure(solve, q, k, page_size=16, page_budget=1):
    before_q, before_k = q.clone(), k.clone()
    got = solve(q, k, page_size, page_budget)
    assert isinstance(got, dict) and set(got) == KEYS, f"键不对：{sorted(got)}"
    assert torch.equal(q, before_q) and torch.equal(k, before_k), "输入被就地修改了"
    num_heads, head_dim = int(q.shape[1]), int(q.shape[2])
    num_pages = -(-int(k.shape[0]) // page_size)
    keep = min(page_budget, num_pages) - 1
    assert got["sign"].shape == (num_heads, head_dim)
    assert got["page_max"].shape == (num_heads, num_pages, head_dim)
    assert got["scores"].shape == (num_heads, num_pages - 1)
    assert got["selected"].shape == (num_heads, keep + 1)
    assert got["selected"].dtype == torch.int64, got["selected"].dtype
    for key in ("sign", "page_max", "scores"):
        assert got[key].dtype == q.dtype, f"{key} dtype {got[key].dtype} != {q.dtype}"
        assert torch.isfinite(got[key]).all(), f"{key} 出现非有限值"
    assert set(got["sign"].unique().tolist()) <= {1.0, -1.0}, "sign 只能是 ±1"
    for row in got["selected"]:
        idx = row.tolist()
        assert idx == sorted(idx), f"selected 必须升序：{idx}"
        assert len(set(idx)) == len(idx), f"selected 有重复页号：{idx}"
        assert idx[-1] == num_pages - 1, f"最后一页必须恒载：{idx}"
        assert all(0 <= i < num_pages for i in idx), f"页号越界：{idx}"
    return got


def _verify(solve, q, k, page_size=16, page_budget=1):
    got = _check_structure(solve, q, k, page_size, page_budget)
    want = _oracle(q, k, page_size, page_budget)
    for key in ("sign", "page_max", "scores"):
        _close(got[key], want[key])
    assert got["selected"].tolist() == want["selected"].tolist(), (
        f"selected：{got['selected'].tolist()} vs {want['selected'].tolist()}")
    return got


def check_example(solve):
    """题面示例：(Lq,H,D)=(1,1,2)、Lkv=6、page_size=2、page_budget=2。"""
    q = torch.tensor([[[1.0, -2.0]]])
    k = torch.tensor([[[1.0, 3.0]], [[2.0, 1.0]], [[-1.0, -0.5]],
                      [[0.5, -0.2]], [[9.0, 0.5]], [[9.0, -1.0]]])
    got = _verify(solve, q, k, 2, 2)
    assert got["sign"].tolist() == [[1.0, -1.0]]
    assert got["page_max"].tolist() == [[[2.0, -1.0], [0.5, 0.5], [9.0, 1.0]]]
    assert got["scores"].tolist() == [[0.0, 1.5]], got["scores"].tolist()
    assert got["selected"].tolist() == [[1, 2]], got["selected"].tolist()


check_example.example = True


def check_zero_query_goes_negative(solve):
    """q_d == 0 必须落到负号支：sign=-1、page_max 取到 min(k) 那一端。"""
    q = torch.tensor([[[0.0, 3.0]]])
    k = torch.tensor([[[5.0, 1.0]], [[7.0, 2.0]], [[1.0, 4.0]], [[9.0, 0.5]]])
    got = _verify(solve, q, k, 2, 2)
    assert got["sign"].tolist() == [[-1.0, 1.0]], "零值必须归到 -1，不能用 torch.sign 得 0"
    ch0 = got["page_max"][0, :, 0].tolist()
    assert ch0 == [-5.0, -1.0], f"通道 0 应取页内 -k 的 amax，got {ch0}"
    assert torch.all(got["sign"] != 0), "sign 不允许出现 0"


def check_zero_sign_changes_page_extreme(solve):
    """零值归负会让 page_max 走到 -k 的 amax 一端；若按 +1 处理则取到另一端的值。"""
    q = torch.tensor([[[0.0, 1.0]]])
    k = torch.tensor([[[-4.0, 1.0]], [[2.0, 1.0]], [[8.0, 1.0]], [[8.0, 1.0]]])
    got = solve(q, k, 2, 3)
    assert got["sign"][0, 0].item() == -1.0
    # 通道 0 取负号 → 页 0 内 max(-k) = max(4, -2) = 4；若错成 +1 会得到 2
    assert got["page_max"][0, 0, 0].item() == 4.0, got["page_max"].tolist()
    assert got["scores"].tolist() == [[1.0]], got["scores"].tolist()


def check_last_page_is_loaded_but_never_scored(solve):
    """末页上界最高也必须从 scores 里消失，但永远出现在 selected 末尾。"""
    q = torch.tensor([[[1.0]]])
    k = torch.tensor([[[0.1]], [[0.2]], [[0.3]], [[0.4]], [[100.0]], [[100.0]]])
    for budget in (1, 2, 3, 4, 10):
        got = _verify(solve, q, k, 2, budget)
        num_pages = 3
        assert got["scores"].shape == (1, num_pages - 1), "末页不能进 scores"
        keep = min(budget, num_pages) - 1
        assert got["selected"].shape == (1, keep + 1)
        assert got["selected"][0, -1].item() == num_pages - 1


def check_budget_caps_at_page_count(solve):
    """page_budget >= P 等价于全加载；page_budget=1 只留末页。"""
    torch.manual_seed(7)
    k = torch.randn(10, 2, 3)
    q = torch.randn(1, 2, 3)
    full = _verify(solve, q, k, 2, 5)
    assert full["selected"].tolist() == [[0, 1, 2, 3, 4]] * 2
    huge = _verify(solve, q, k, 2, 99)
    assert huge["selected"].tolist() == full["selected"].tolist(), "超预算应截断到 P"
    one = _verify(solve, q, k, 2, 1)
    assert one["selected"].tolist() == [[4]] * 2, one["selected"].tolist()


def check_ties_prefer_smaller_page(solve):
    """所有页同分时，按更小的页号取。"""
    k = torch.ones(8, 1, 2)
    q = torch.tensor([[[1.0, 2.0]]])            # P = 4，三页竞标、取 2 个
    got = _verify(solve, q, k, 2, 3)
    assert got["scores"].tolist() == [[3.0, 3.0, 3.0]], got["scores"].tolist()
    assert got["selected"].tolist() == [[0, 1, 3]], got["selected"].tolist()


def check_pad_sentinel_does_not_floor_negative_pages(solve):
    """整页全负的通道：amax 必须留在负值，不能被补齐哨兵抬到 0。"""
    q = torch.tensor([[[1.0]]])
    k = torch.tensor([[[-1.0]], [[-2.0]], [[-3.0]]])          # Lkv=3 -> P=2，末页补 1 个哨兵
    got = _verify(solve, q, k, 2, 2)
    assert got["page_max"][0, 1, 0].item() == -3.0, got["page_max"].tolist()
    assert got["scores"][0, 0].item() == -1.0, got["scores"].tolist()


def check_score_is_an_upper_bound(solve):
    """scores[h,p] 必须 ≥ 该页任一 token 的真实 q·k，且至少有一页打到端点。"""
    for seed in range(6):
        torch.manual_seed(seed)
        page_size, kv_len, heads, dim = 4, 13, 3, 5
        num_pages = -(-kv_len // page_size)
        k = torch.randn(kv_len, heads, dim, dtype=torch.float64) * 2.0
        q = torch.randn(1, heads, dim, dtype=torch.float64)
        got = solve(q, k, page_size, num_pages)
        for p in range(num_pages - 1):
            lo, hi = p * page_size, min((p + 1) * page_size, kv_len)
            true = (k[lo:hi] * q[0].unsqueeze(0)).sum(dim=-1)   # (n, H) 真实 q·k
            assert (got["scores"][:, p] >= true.max(dim=0).values - 1e-9).all(), (
                f"seed={seed} page={p} 上界穿底了")
        assert got["scores"].dtype == torch.float64
        assert torch.equal(got["sign"] > 0, q[0] > 0), "sign 的符号位应与 q>0 完全一致"


def check_scores_are_raw_dot_products(solve):
    """没有 softmax / 没有 1/sqrt(d)：整体放大 10 倍，分数也放大 10 倍。"""
    torch.manual_seed(13)
    k = torch.randn(12, 2, 4, dtype=torch.float64)
    q = torch.randn(1, 2, 4, dtype=torch.float64)
    a = solve(q, k, 3, 3)
    b = solve(q * 10.0, k, 3, 3)
    assert torch.allclose(b["scores"], 10.0 * a["scores"], rtol=1e-12, atol=1e-12)
    assert torch.all(torch.isfinite(a["scores"]))


def check_non_contiguous_slice(solve):
    """从更大的张量里切出来的非连续视图必须和 contiguous 副本一致。"""
    torch.manual_seed(17)
    big = torch.randn(20, 2, 4, dtype=torch.float64)
    k = big[::2]                                  # 步长 2，非连续
    q = torch.randn(1, 2, 4, dtype=torch.float64)[:, :, :]
    a = solve(q, k, 2, 2)
    b = solve(q.contiguous(), k.contiguous(), 2, 2)
    for key in ("sign", "page_max", "scores"):
        _close(a[key], b[key])
    assert a["selected"].tolist() == b["selected"].tolist()


def check_dtype_sweep(solve):
    """float32 输入的输出必须是 float32；数值与 float64 结果在 1e-5 内一致。"""
    torch.manual_seed(19)
    hi_k = torch.randn(10, 2, 3, dtype=torch.float64)
    hi_q = torch.randn(1, 2, 3, dtype=torch.float64)
    lo = solve(hi_q.float(), hi_k.float(), 2, 3)
    hi = solve(hi_q, hi_k, 2, 3)
    for key in ("sign", "page_max", "scores"):
        assert lo[key].dtype == torch.float32, f"{key} 应为 float32"
        torch.testing.assert_close(lo[key], hi[key].float(), rtol=1e-5, atol=1e-5)
    assert lo["selected"].dtype == torch.int64


def check_random_against_oracle(solve):
    """随机形状与 numpy oracle 全键对拍。"""
    cases = [(1, 2, 1, 2), (16, 33, 1, 4), (5, 17, 4, 8), (2, 9, 2, 16),
             (7, 64, 3, 5), (3, 100, 2, 7)]
    for seed, (page_size, kv_len, heads, dim) in enumerate(cases):
        torch.manual_seed(100 + seed)
        k = torch.randn(kv_len, heads, dim, dtype=torch.float64)
        q = torch.randn(1, heads, dim, dtype=torch.float64)
        num_pages = -(-kv_len // page_size)
        for budget in (1, 2, num_pages // 2 + 1, num_pages, num_pages + 3):
            _verify(solve, q, k, page_size, max(1, budget))


def check_value_errors(solve):
    q = torch.randn(1, 2, 3)
    k = torch.randn(8, 2, 3)
    bad = [
        (torch.randn(2, 2, 3), k, 2, 2),                      # Lq != 1
        (q, torch.randn(2, 2, 3), 2, 2),                      # P < 2
        (q, torch.randn(8, 3, 3), 2, 2),                      # head 数不一致
        (q, torch.randn(8, 2, 4), 2, 2),                      # head_dim 不一致
        (q, k.double(), 2, 2),                                # dtype 不同
        (q.to(torch.int32), k.to(torch.int32), 2, 2),         # 非浮点
        (q.reshape(1, 6), k, 2, 2),                           # 非三维
        (q, k, 0, 2),                                         # page_size < 1
        (q, k, 2, 0),                                         # page_budget < 1
        (q, k, 2.0, 2),                                       # page_size 非整数
        (q, k, 2, True),                                      # bool 不算 int
    ]
    for i, args in enumerate(bad):
        try:
            solve(*args)
        except ValueError:
            continue
        raise AssertionError(f"第 {i} 个非法输入应抛 ValueError：{[getattr(a, 'shape', a) for a in args]}")
