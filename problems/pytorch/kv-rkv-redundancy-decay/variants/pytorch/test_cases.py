import numpy as np
import torch

ORDER = ("attn", "redundancy", "final", "kept", "kept_keys", "kept_values")


def _oracle(k, q, v, budget, window, kernel, lam, thr, direction):
    """numpy 独立重算：softmax→窗口均值→max_pool；余弦去对角→代表位清零→列均值→softmax。"""
    kn, vn = k.numpy().astype(float), v.numpy().astype(float)                # (L,H,D)
    kv_len, heads, dim = kn.shape
    n = kv_len - window
    logits = np.einsum("whd,lhd->hwl", q.numpy().astype(float), kn[:n]) / np.sqrt(dim)
    prob = np.exp(logits - logits.max(-1, keepdims=True))
    raw = (prob / prob.sum(-1, keepdims=True)).mean(1)                       # (H, n)
    half = kernel // 2
    padded = np.full((heads, n + 2 * half), -np.inf)
    padded[:, half:half + n] = raw
    attn = np.lib.stride_tricks.sliding_window_view(padded, kernel, -1).max(-1)
    unit = kn / (np.linalg.norm(kn, axis=-1, keepdims=True) + 1e-8)
    sim = np.einsum("lhd,mhd->hlm", unit, unit)
    sim[:, np.arange(kv_len), np.arange(kv_len)] = 0.0
    idx = np.where(sim > thr, np.arange(kv_len), 0)                          # 上游补 0 下标行
    rep = idx.max(-1) if direction == "last" else idx.min(-1)
    sim[np.arange(heads)[:, None], np.arange(kv_len), rep] = 0.0
    col = sim.mean(-2)
    weight = np.exp(col - col.max(-1, keepdims=True))
    redundancy = (weight / weight.sum(-1, keepdims=True))[:, :n]
    final = lam * attn - (1.0 - lam) * redundancy
    order = np.argsort(-final, axis=-1, kind="stable")                       # 同分保留小下标
    kept = np.concatenate([order[:, : budget - window], np.tile(np.arange(n, kv_len), (heads, 1))], -1)
    return {"attn": attn, "redundancy": redundancy, "final": final, "kept": kept,
            "kept_keys": np.take_along_axis(kn.transpose(1, 0, 2), kept[:, :, None], 1),
            "kept_values": np.take_along_axis(vn.transpose(1, 0, 2), kept[:, :, None], 1)}


def _close(got, want, tol=2e-9):
    torch.testing.assert_close(torch.as_tensor(got, dtype=torch.float64),
                               torch.as_tensor(want, dtype=torch.float64), rtol=tol, atol=tol)


def _case(seed, kv_len, heads, dim, window):
    """固定种子的随机用例：(k, q, v)，v 只用来核对 gather 顺序。"""
    torch.manual_seed(seed)
    return (torch.randn(kv_len, heads, dim, dtype=torch.float64),
            torch.randn(window, heads, dim, dtype=torch.float64),
            torch.arange(kv_len * heads * dim, dtype=torch.float64).reshape(kv_len, heads, dim))


def _verify(solve, k, q, v, **kw):
    """结构检查（键 / 形状 / dtype / kept 语义）+ numpy oracle 全键对拍。"""
    before = (k.clone(), q.clone(), v.clone())
    got = solve(k, q, v, **kw)
    window, budget = kw.get("window_size", 8), kw.get("budget", 128)
    n, heads, dim = int(k.shape[0]) - window, int(k.shape[1]), int(k.shape[2])
    tol = 3e-5 if k.dtype == torch.float32 else 2e-9
    assert set(got) == set(ORDER), f"键应为 {sorted(ORDER)}，got {sorted(got)}"
    assert all(torch.equal(a, b) for a, b in zip((k, q, v), before)), "输入被就地修改了"
    shape = {"attn": (heads, n), "redundancy": (heads, n), "final": (heads, n), "kept": (heads, budget),
             "kept_keys": (heads, budget, dim), "kept_values": (heads, budget, dim)}
    for key in ORDER:
        assert tuple(got[key].shape) == shape[key], f"{key} 形状应为 {shape[key]}，got {tuple(got[key].shape)}"
        assert key == "kept" or got[key].dtype == k.dtype, f"{key} 应沿用输入 dtype {k.dtype}"
    assert got["kept"].dtype == torch.int64, "kept 必须是 int64"
    for row in got["kept"].tolist():
        assert len(set(row)) == len(row), f"kept 不能有重复：{row}"
        assert row[-window:] == list(range(n, n + window)), f"末尾 window_size 位原序恒留：{row}"
    want = _oracle(k, q, v, budget, window, kw.get("kernel_size", 7), kw.get("mix_lambda", 0.07),
                   kw.get("threshold", 0.5), kw.get("retain_direction", "last"))
    for key in ORDER[1:4] + ORDER[4:]:
        _close(got[key], want[key], tol)
    assert got["kept"].tolist() == want["kept"].tolist(), (
        f"kept 应为前 budget-window_size 条按 final 降序（同分取小下标）再拼窗口：{got['kept'].tolist()}")
    return got


def check_example(solve):
    """题面示例：L=4、H=1、D=2、window_size=1、kernel_size=1、budget=3。"""
    k = torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]]])
    q, v = torch.tensor([[[1.0, 0.0]]]), torch.tensor([[[0.0, 1.0]], [[2.0, 3.0]], [[4.0, 5.0]], [[6.0, 7.0]]])
    got = _verify(solve, k, q, v, budget=3, window_size=1, kernel_size=1)
    for key, want in (("attn", [0.4011, 0.4011, 0.1978]), ("redundancy", [0.3026, 0.3026, 0.1975]),
                      ("final", [-0.2533, -0.2533, -0.1698])):
        _close(got[key], [want], 1e-4)
    assert got["kept"].tolist() == [[2, 0, 3]], f"应留下 [2, 0, 3]，got {got['kept'].tolist()}"


check_example.example = True


def check_attn_excludes_window_keys(solve):
    """attn 只看前 L-window 个 key；改窗口内 key 只能改 redundancy。"""
    k, q, v = _case(1, 6, 1, 3, 2)
    a = _verify(solve, k, q, v, budget=5, window_size=2, kernel_size=1)
    k2 = k.clone()
    k2[4:] += 3.0
    b = _verify(solve, k2, q, v, budget=5, window_size=2, kernel_size=1)
    _close(a["attn"], b["attn"])
    _close(a["attn"].sum(-1), [1.0])
    assert not torch.allclose(a["redundancy"], b["redundancy"]), "改窗口内的 key 必须改变 redundancy"


def check_redundancy_softmax_counts_window(solve):
    """redundancy 先在全长 L 上 softmax 再切窗口列，切片后行和必须 < 1。"""
    k, q, v = _case(2, 7, 1, 2, 1)
    k[6] = k[0]
    got = _verify(solve, k, q, v, budget=4, window_size=1, kernel_size=1)
    assert got["redundancy"].sum().item() < 0.999, "分母含窗口列，切片后行和应明显小于 1"
    assert (got["redundancy"] > 0).all(), "冗余分是 softmax 输出，必须全为正"


def check_empty_hit_row_zeroes_column_zero(solve):
    """没有 >threshold 的邻居时 max(下标) 返回 0，0 号列的相似度照样被清掉。"""
    k = torch.stack([torch.tensor([1.0, 0.0], dtype=torch.float64),
                     torch.tensor([0.4, np.sqrt(0.84)], dtype=torch.float64)]).unsqueeze(1)
    got = _verify(solve, k, torch.tensor([[[1.0, 0.0]]], dtype=torch.float64),
                  torch.zeros(2, 1, 2, dtype=torch.float64), budget=2, window_size=1, kernel_size=1)
    expect = 1.0 / (1.0 + np.exp(0.2))                                    # 两条 key 余弦 0.4
    _close(got["redundancy"], [[expect]], 1e-6)
    assert expect < 0.499, "0 号列被清 0 后应低于对称情形下的 0.5"


def check_threshold_is_strict(solve):
    """余弦恰好等于 threshold 不算冗余（上游是严格 >）。"""
    k = torch.tensor([[[1.0, 0.0]], [[0.5, np.sqrt(0.75)]]], dtype=torch.float64)   # 余弦 0.5
    got = _verify(solve, k, torch.tensor([[[1.0, 0.0]]], dtype=torch.float64),
                  torch.zeros(2, 1, 2, dtype=torch.float64), budget=2, window_size=1, kernel_size=1)
    _close(got["redundancy"], [[1.0 / (1.0 + np.exp(0.25))]], 1e-6)
    assert got["redundancy"][0, 0].item() < 0.45, "0.5 不算命中时首项应约 0.4378"


def check_duplicate_pair_loses_to_unique(solve):
    """重复对互相抬高低冗余分；mix_lambda=0.07 时冗余主导，重复对双双让位。"""
    k, q, v = _case(3, 4, 1, 2, 1)
    k[1] = k[0]
    got = _verify(solve, k, q, v, budget=3, window_size=1, kernel_size=1)
    assert got["kept"][0, :2].tolist() != [0, 1], "k0/k1 重复，不能两条一起进前列"
    pure = _verify(solve, k, q, v, budget=3, window_size=1, kernel_size=1, mix_lambda=1.0)
    assert pure["kept"][0, 1].item() == 1, "λ=1 只剩注意力时重复对都进前列"


def check_mix_lambda_weights(solve):
    """final 必须严格等于 mix_lambda*attn - (1-mix_lambda)*redundancy。"""
    k, q, v = _case(4, 5, 2, 3, 1)
    for lam in (0.0, 0.07, 0.5, 1.0):
        got = _verify(solve, k, q, v, budget=3, window_size=1, kernel_size=3, mix_lambda=lam)
        _close(got["final"], lam * got["attn"] - (1.0 - lam) * got["redundancy"])
    zero = solve(k, q, v, 3, 1, 3, 0.0)
    _close(zero["final"], -zero["redundancy"])


def check_retain_direction_first(solve):
    """first 取的是补 0 下标行的 min，恒为 0：只剩 0 号列被整列清掉。"""
    k, q, v = _case(5, 6, 1, 2, 1)
    k[1], k[3], k[5] = k[0].clone(), k[0].clone(), k[0].clone()
    last = _verify(solve, k, q, v, budget=4, window_size=1, kernel_size=1)
    first = _verify(solve, k, q, v, budget=4, window_size=1, kernel_size=1, retain_direction="first")
    none = _verify(solve, k, q, v, budget=4, window_size=1, kernel_size=1, threshold=1.0)
    assert not torch.allclose(last["redundancy"], first["redundancy"]), (
        "last 清每行最大命中下标；first 的 min 落在补 0 位，只剩 0 号列被清")
    _close(first["redundancy"], none["redundancy"])      # 等价于「没有命中、每行都清 0 号列」


def check_budget_edges(solve):
    """cache_len == budget 时一条不丢；budget 只比 window 大 1 时只留一条旧 token。"""
    k, q, v = _case(6, 8, 2, 3, 3)
    full = _verify(solve, k, q, v, budget=8, window_size=3, kernel_size=3)
    assert set(full["kept"][0].tolist()) == set(range(8)), "预算等于长度时一条都不能丢"
    tight = _verify(solve, k, q, v, budget=4, window_size=3, kernel_size=3)
    assert [row.tolist()[-3:] for row in tight["kept"]] == [[5, 6, 7]] * 2


def check_ties_prefer_smaller_index(solve):
    """全同分时按更小的下标取（上游 torch.topk 不保证同分顺序，这里钉死）。"""
    got = _verify(solve, torch.ones(5, 1, 2), torch.tensor([[[1.0, 0.0]]]), torch.zeros(5, 1, 2),
                  budget=3, window_size=1, kernel_size=1, mix_lambda=1.0)
    _close(got["final"], [[0.25, 0.25, 0.25, 0.25]], 1e-5)
    assert got["kept"].tolist() == [[0, 1, 4]], f"同分取小下标，应得 [0, 1, 4]，got {got['kept'].tolist()}"


def check_dtype_and_no_mutation(solve):
    """float32 输入 → float32 输出；余弦矩阵必须另开张量，不能改 k/q/v。"""
    k, q, v = _case(7, 9, 2, 4, 2)
    lo = (k.clone().float(), q.clone().float(), v.clone().float())
    before = (k.clone(), q.clone(), v.clone())
    got = solve(k, q, v, 5, 2, 3)
    assert all(torch.equal(a, b) for a, b in zip((k, q, v), before)), "输入被就地修改了"
    for key in ("attn", "redundancy", "final", "kept_keys", "kept_values"):
        assert got[key].dtype == torch.float64, f"{key} 应为 float64"
    small = solve(*lo, 5, 2, 3)
    for key in ("attn", "redundancy", "final", "kept_keys", "kept_values"):
        assert small[key].dtype == torch.float32, f"{key} 应沿用 float32 输入"
    assert small["kept"].dtype == torch.int64, "kept 必须是 int64"
    torch.testing.assert_close(small["final"], got["final"].float(), rtol=1e-5, atol=1e-5)
    assert small["kept"].tolist() == got["kept"].tolist(), "float32/float64 的选点必须一致"


def check_random_against_oracle(solve):
    """多组形状 / 窗口 / 卷积核 / 预算与 numpy oracle 全键对拍。"""
    cases = [(4, 1, 3, 2, 1), (6, 2, 2, 1, 3), (10, 3, 4, 2, 5), (12, 2, 3, 4, 3), (16, 4, 2, 8, 7)]
    for seed, (kv_len, heads, dim, window, kernel) in enumerate(cases):
        k, q, v = _case(50 + seed, kv_len, heads, dim, window)
        for budget in (window + 1, max(window + 1, (kv_len - window) // 2 + 1), kv_len):
            _verify(solve, k, q, v, budget=budget, window_size=window, kernel_size=kernel)


def check_value_errors(solve):
    """非法入参一律 ValueError；cache_len == budget 这类边界必须能跑。"""
    k, q, v = torch.randn(8, 2, 3), torch.randn(2, 2, 3), torch.zeros(8, 2, 3)
    assert solve(k, q, v, budget=4, window_size=2)["kept"].shape == (2, 4), "cache_len==budget 应可跑"
    bad = [(k, q, v, dict(budget=9, window_size=2)),                 # cache_len < budget
           (k, q, v, dict(budget=2, window_size=2)),                 # budget 未超过 window
           (k, q, v, dict(budget=4, window_size=2, kernel_size=4)),  # 偶数核
           (k, q, v, dict(budget=4, window_size=2, retain_direction="mid")),
           (k, q, v, dict(budget=4, window_size=2, mix_lambda=1.5)),
           (k, q, v, dict(budget=4.0, window_size=2)),               # budget 非 int
           (k, k, v, dict(budget=4, window_size=2)),                 # q 行数 != window
           (k[:, :1], q, v, dict(budget=4, window_size=2)),          # head 数不一致
           (k, q, k.double(), dict(budget=4, window_size=2)),        # dtype 不一致
           (k.reshape(8, 6), q, v, dict(budget=4, window_size=2))]   # 非三维
    for i, (kk, qq, vv, kw) in enumerate(bad):
        try:
            solve(kk, qq, vv, **kw)
        except ValueError:
            continue
        raise AssertionError(f"第 {i} 个非法输入应抛 ValueError：{kw}")
