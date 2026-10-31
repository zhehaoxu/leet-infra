import numpy as np
import torch

KEYS = {"first_hop", "query_weight", "kept_query", "score", "selected"}


def _rand_case(seed, seq, heads, kv, dim, req=3):
    torch.manual_seed(seed)
    q = torch.randn(seq, heads, dim, dtype=torch.float64)
    k = torch.randn(seq, kv, dim, dtype=torch.float64)
    mask = torch.zeros(seq, dtype=torch.bool)
    mask[seq - req:] = True
    return q, k, mask


def _oracle(q, k, request_mask, window, tau, recent, budget):
    """numpy 独立重算：因果 softmax → 组内求和 → 权重/阈值 → 二跳 → 尾部恒保。"""
    qn, kn = q.detach().numpy().astype(np.float64), k.detach().numpy().astype(np.float64)
    seq, heads, dim = qn.shape
    num_kv, group = kn.shape[1], heads // kn.shape[1]
    mask = np.asarray(request_mask.detach().numpy(), dtype=bool)
    logits = np.einsum("thd,ihd->hti", qn, np.repeat(kn, group, axis=1)) * dim ** -0.5
    logits = np.where(np.triu(np.ones((seq, seq), dtype=bool), 1)[None], -np.inf, logits)
    logits = logits - logits.max(axis=-1, keepdims=True)
    prob = np.exp(logits)
    grouped = (prob / prob.sum(axis=-1, keepdims=True)).reshape(num_kv, group, seq, seq).sum(axis=1)
    first = grouped[:, seq - min(window, seq):, :].sum(axis=1)
    raw = first.copy()
    raw[:, mask] = 1.0
    weight = raw / np.maximum(raw.sum(axis=-1, keepdims=True), 1e-12)
    if tau > 0:
        keep = weight >= tau / seq
        keep[:, mask] = True
        weight = weight * keep
    else:
        keep = np.ones_like(weight, dtype=bool)
    score = np.einsum("kq,kqi->ki", weight, grouped)
    head, slots, rows = seq - recent, budget - recent, []
    for kv in range(num_kv):
        picked = sorted(range(head), key=lambda i: (-score[kv, i], i))[:slots]
        rows.append(sorted(picked + list(range(head, seq))))
    return {
        "first_hop": torch.tensor(first, dtype=q.dtype),
        "query_weight": torch.tensor(weight, dtype=q.dtype),
        "kept_query": torch.tensor(np.flatnonzero(keep.any(axis=0)), dtype=torch.int64),
        "score": torch.tensor(score, dtype=q.dtype),
        "selected": torch.tensor(np.array(rows, dtype=np.int64).reshape(num_kv, budget)),
    }


def _close(a, b):
    tol = 3e-5 if a.dtype == torch.float32 else 2e-9
    torch.testing.assert_close(a, b, rtol=tol, atol=tol)


def _verify(solve, q, k, mask, window=8, tau=1.0, recent=4, budget=12):
    """结构 + numpy oracle 全键对拍。"""
    saved = (q.clone(), k.clone())
    got = solve(q, k, mask, window, tau, recent, budget)
    assert torch.equal(q, saved[0]) and torch.equal(k, saved[1]), "输入被就地修改了"
    assert isinstance(got, dict) and set(got) == KEYS, f"键不对：{sorted(got)}"
    seq, num_kv = int(q.shape[0]), int(k.shape[1])
    for key, shape in (("first_hop", (num_kv, seq)), ("score", (num_kv, seq)),
                       ("query_weight", (num_kv, seq)), ("selected", (num_kv, budget))):
        assert got[key].shape == shape, f"{key} 形状 {tuple(got[key].shape)} != {shape}"
    for key in ("first_hop", "query_weight", "score"):
        assert got[key].dtype == q.dtype, f"{key} dtype {got[key].dtype} != {q.dtype}"
        assert torch.isfinite(got[key]).all() and (got[key] >= 0).all(), f"{key} 必须有限且非负"
    assert got["kept_query"].dtype == torch.int64 and got["selected"].dtype == torch.int64
    kept = got["kept_query"].tolist()
    assert kept == sorted(set(kept)), f"kept_query 必须升序无重复：{kept}"
    for row in got["selected"]:
        idx = row.tolist()
        assert idx == sorted(set(idx)) and all(0 <= i < seq for i in idx), f"selected 非法：{idx}"
        assert idx[budget - recent:] == list(range(seq - recent, seq)), f"尾部必须恒保：{idx}"
    want = _oracle(q, k, mask, window, tau, recent, budget)
    for key in ("first_hop", "query_weight", "score"):
        _close(got[key], want[key])
    mass = torch.full((num_kv,), float(q.shape[1] * window // num_kv), dtype=q.dtype)
    _close(got["first_hop"].sum(dim=-1), mass)          # 因果 attention 每行质量为 1
    assert kept == want["kept_query"].tolist(), "kept_query 必须是各头保留 query 的并集"
    assert got["selected"].tolist() == want["selected"].tolist(), (
        f"selected：{got['selected'].tolist()} vs {want['selected'].tolist()}")
    return got


def check_example(solve):
    """题面示例：N=6、H=KV=1、D=1、k 全 0（注意力均匀），window=2、tau=1、recent=1、budget=3。"""
    mask = torch.tensor([False] * 4 + [True] * 2)
    got = _verify(solve, torch.ones(6, 1, 1, dtype=torch.float64),
                  torch.zeros(6, 1, 1, dtype=torch.float64), mask, 2, 1.0, 1, 3)
    f64 = torch.float64
    _close(got["first_hop"][0], torch.tensor([11 / 30] * 5 + [1 / 6], dtype=f64))
    _close(got["query_weight"][0], torch.tensor([0.0] * 4 + [30 / 104] * 2, dtype=f64))
    assert got["kept_query"].tolist() == [4, 5], "11/104 < 1/6，前四个 query 必须被剔除"
    _close(got["score"][0], torch.tensor([11 / 104] * 5 + [5 / 104], dtype=f64))
    assert got["selected"].tolist() == [[0, 1, 5]], "五个同分位置取最小的两个"


check_example.example = True


def check_request_queries_get_raw_weight_one(solve):
    """请求段归一化前是字面的 1：w_req = 1 / (Σ_{j∉request} s(j) + n_req)，且 Σw = 1。"""
    q, k, mask = _rand_case(29, 24, 4, 2, 6, req=4)
    got = _verify(solve, q, k, mask, 8, 0.0, 4, 12)
    inv = 1.0 / (got["first_hop"][:, ~mask].sum(dim=1) + 4)
    want = got["first_hop"] * inv[:, None]
    want[:, mask] = inv[:, None].expand(2, 4)
    _close(got["query_weight"], want)
    _close(got["query_weight"].sum(dim=-1), torch.ones(2, dtype=torch.float64))


def check_threshold_cuts_at_tau_over_n(solve):
    """阈值是 tau/N（N 为总长）：恰好 w >= tau/N 的保留，差一点点的被置 0。"""
    q, k, mask = _rand_case(23, 20, 2, 1, 5, req=3)
    pre = _verify(solve, q, k, mask, 6, 0.0, 3, 9)["query_weight"][0]
    doc = sorted(float(pre[i]) for i in range(17))
    tau = (doc[4] + doc[5]) / 2.0 * 20                     # 实现看到的是 tau / 20
    got = _verify(solve, q, k, mask, 6, tau, 3, 9)
    expect = torch.tensor([1.0 if (i >= 17 or float(pre[i]) >= tau / 20) else 0.0 for i in range(20)],
                          dtype=torch.float64)
    assert int(expect.sum()) == 15, "自检：应有 5 个文档 query 被剔除"
    assert torch.equal(got["query_weight"][0] > 0, expect > 0), "被剔除的条件是 w < tau/N"
    _close(got["query_weight"][0], pre * expect)


def check_all_non_request_queries_dropped(solve):
    """tau 极大时非请求 query 全被剔除：w 之和 < 1（归一化发生在剔除之前）。"""
    q, k, mask = _rand_case(30, 16, 2, 1, 4, req=3)
    got = _verify(solve, q, k, mask, 4, 1e9, 2, 7)
    assert got["kept_query"].tolist() == [13, 14, 15], got["kept_query"].tolist()
    assert (got["query_weight"][0, :13] == 0).all(), "非请求 query 必须被置 0"
    total = float(got["first_hop"][0, :13].sum()) + 3
    _close(got["query_weight"].sum(dim=-1), torch.full((1,), 3.0 / total, dtype=torch.float64))


def check_causal_boundary_and_group_sum(solve):
    """请求段只有下标 0 时 score = [G*w(0), 0, ...]：q < i 的行被因果掩码挡住。"""
    q, k, mask = _rand_case(37, 8, 2, 1, 3, req=0)
    mask[0] = True
    got = _verify(solve, q, k, mask, 1, 1e9, 2, 5)
    assert got["kept_query"].tolist() == [0], got["kept_query"].tolist()
    w0 = float(got["query_weight"][0, 0])
    _close(got["score"][0, 0], torch.tensor(2.0 * w0, dtype=torch.float64))
    assert (got["score"][0, 1:] == 0).all(), "没有 q >= i 的 query 时 score(i) 必须为 0"


def check_tail_is_always_kept(solve):
    """最后 recent 个 key 恒保；竞标名额只来自 [0, N-recent)，名额为 0 时只剩尾部。"""
    q, k, mask = _rand_case(41, 16, 2, 1, 4, req=4)
    for recent, budget in ((4, 6), (4, 5), (4, 4), (1, 9), (1, 15)):
        row = _verify(solve, q, k, mask, 4, 1.0, recent, budget)["selected"][0].tolist()
        assert all(i < 16 - recent for i in row[:budget - recent]), f"竞标名额只能来自前缀：{row}"
    assert _verify(solve, q, k, mask, 4, 1.0, 4, 4)["selected"][0].tolist() == [12, 13, 14, 15]


def check_ties_pick_smaller_index(solve):
    """前 9 个位置同分时取更小的下标（torch.topk 不定义同分顺序，这里钉死）。"""
    mask = torch.zeros(10, dtype=torch.bool)
    mask[8:] = True
    got = _verify(solve, torch.ones(10, 1, 1, dtype=torch.float64),
                  torch.zeros(10, 1, 1, dtype=torch.float64), mask, 2, 1.0, 1, 4)
    _close(got["score"][0], torch.tensor([19 / 332] * 9 + [9 / 332], dtype=torch.float64))
    assert got["selected"].tolist() == [[0, 1, 2, 9]], got["selected"].tolist()


def check_kept_query_is_the_union_over_heads(solve):
    """kept_query 是各 KV 头保留集合的并集：一个头剔除、另一个头保留的 query 仍要算。"""
    q, k, mask = _rand_case(43, 18, 4, 2, 5, req=4)
    for tau in (0.4, 0.8, 1.2, 2.0, 4.0):
        got = _verify(solve, q, k, mask, 6, tau, 3, 8)
        alive = got["query_weight"] > 0
        if alive.any(dim=0).sum() > alive.all(dim=0).sum():
            break
    assert alive.any(dim=0).sum() > alive.all(dim=0).sum(), "自检：两头的保留集合必须不同"
    assert got["kept_query"].tolist() == alive.any(dim=0).nonzero().flatten().tolist(), (
        "kept_query 必须是并集，不是交集或单头结果")


def check_random_against_oracle(solve):
    """随机形状 / 超参与 numpy oracle 全键对拍。"""
    cases = [(24, 4, 2, 8, 5, 2.0, 4, 12), (33, 6, 3, 5, 16, 1.0, 8, 20),
             (17, 2, 1, 4, 1, 0.5, 0, 9), (40, 8, 2, 7, 12, 0.0, 16, 30)]
    for seed, (seq, heads, kv, dim, window, tau, recent, budget) in enumerate(cases):
        _verify(solve, *_rand_case(100 + seed, seq, heads, kv, dim, req=6),
                window, tau, recent, budget)


def check_dtype_sweep(solve):
    """float32 输入的输出必须是 float32，且与 float64 结果在 1e-5 内一致。"""
    q, k, mask = _rand_case(19, 26, 4, 2, 8, req=4)
    lo, hi = solve(q.float(), k.float(), mask, 8, 1.0, 4, 14), solve(q, k, mask, 8, 1.0, 4, 14)
    for key in ("first_hop", "query_weight", "score"):
        assert lo[key].dtype == torch.float32, f"{key} 应为 float32"
        torch.testing.assert_close(lo[key], hi[key].float(), rtol=1e-5, atol=1e-5)
    assert lo["selected"].dtype == torch.int64 and lo["kept_query"].dtype == torch.int64


def check_value_errors(solve):
    """形状 / dtype / 超参非法一律 ValueError。"""
    q, k = torch.randn(10, 2, 3), torch.randn(10, 1, 3)
    m = torch.zeros(10, dtype=torch.bool)
    m[8:] = True
    bad = [(q, torch.randn(9, 1, 3), m, 4, 1.0, 2, 6),      # 序列长度不一致
           (q, torch.randn(10, 1, 4), m, 4, 1.0, 2, 6),     # head_dim 不一致
           (q, k.double(), m, 4, 1.0, 2, 6),                # dtype 不同
           (q.int(), k.int(), m, 4, 1.0, 2, 6),             # 非浮点
           (q, torch.randn(10, 3, 3), m, 4, 1.0, 2, 6),     # H 不是 KV 的倍数
           (q, k, m.long(), 4, 1.0, 2, 6),                  # 掩码非 bool
           (q, k, m[:9], 4, 1.0, 2, 6),                     # 掩码长度不对
           (q, k, m, 0, 1.0, 2, 6),                         # window < 1
           (q, k, m, 11, 1.0, 2, 6),                        # window > N
           (q, k, m, 4, -1.0, 2, 6),                        # tau < 0
           (q, k, m, 4, 1.0, 6, 5),                         # recent > budget
           (q, k, m, 4, 1.0, 2, 10),                        # budget >= N
           (q, k, m, 4, 1.0, 2, 0),                         # budget < 1
           (q[None], k, m, 4, 1.0, 2, 6),                   # 非三维
           (q, k, m, 4.0, 1.0, 2, 6)]                       # window 非整数
    for i, args in enumerate(bad):
        try:
            solve(*args)
        except ValueError:
            continue
        raise AssertionError(f"第 {i} 个非法输入应抛 ValueError")
