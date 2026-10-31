import torch

KEYS = {"tsp_indices", "hidden_out", "position_out", "capacities", "kept"}


def _pick(pairs, k):
    """分数降序、同分取更小下标，返回升序下标。"""
    return sorted(i for i, _ in sorted(pairs, key=lambda p: (-p[1], p[0]))[:k])


def _oracle(caches, q_len, tsp_idx, hidden, pos, window, retain_rate, tsp_rate):
    """纯 python 独立重算：逐层容量、逐 head 保留、TSP 跨 head 求和。"""
    hkv, tsp_length = caches[0].shape[1], int(q_len * tsp_rate)
    gating = q_len > tsp_length
    capacities, kept, tsp = [], [], None
    for layer in range(len(caches)):
        rate = retain_rate if layer <= tsp_idx else retain_rate / tsp_rate
        length = q_len if (layer <= tsp_idx or not gating) else tsp_length
        capacity = int(length * rate)
        capacities.append(capacity)
        scores = caches[layer].tolist()
        if length < capacity:
            kept.append([list(range(length))] * hkv)
            continue
        past = scores[:length - window]
        kept.append([_pick([(i, past[i][h]) for i in range(len(past))], capacity - window)
                     + list(range(length - window, length)) for h in range(hkv)])
        if layer == tsp_idx and gating:
            total = list(enumerate(sum(row) for row in past))
            tsp = _pick(total, tsp_length - window) + list(range(q_len - window, q_len))
    if tsp is None:
        return {"tsp_indices": None, "hidden_out": None, "position_out": None,
                "capacities": capacities, "kept": kept}
    return {"tsp_indices": tsp, "hidden_out": hidden[tsp].tolist(),
            "position_out": pos[tsp].tolist(), "capacities": capacities, "kept": kept}


def _case(seed, q_len, hkv, tsp_idx, layers, window, retain_rate, tsp_rate, dtype=torch.float64):
    """造输入：TSP 层之后的 cache 只有 tsp_length 个 token。"""
    torch.manual_seed(seed)
    tsp_length = int(q_len * tsp_rate)
    lengths = [q_len if (l <= tsp_idx or q_len <= tsp_length) else tsp_length for l in range(layers)]
    caches = [torch.randn(n, hkv, dtype=torch.float64).to(dtype) for n in lengths]
    hidden = torch.stack([torch.arange(q_len), torch.arange(100, 100 + q_len)], 1).to(dtype)
    return [caches, q_len, tsp_idx, hidden, torch.arange(q_len, dtype=torch.int64),
            window, retain_rate, tsp_rate]


def _verify(solve, caches, q_len, tsp_idx, hidden, pos, window, retain_rate, tsp_rate):
    before = [c.clone() for c in caches], hidden.clone(), pos.clone()
    got = solve([c.clone() for c in caches], q_len, tsp_idx, hidden.clone(), pos.clone(),
                window, retain_rate, tsp_rate)
    want = _oracle(caches, q_len, tsp_idx, hidden, pos, window, retain_rate, tsp_rate)
    assert isinstance(got, dict) and set(got) == KEYS, f"键不对：{sorted(got)}"
    assert all(torch.equal(a, b) for a, b in zip(before[0], caches)), "attn_caches 被就地修改了"
    assert torch.equal(before[1], hidden) and torch.equal(before[2], pos), "hidden/position_ids 被就地修改了"
    assert got["capacities"].dtype == torch.int64, "capacities 必须是 int64"
    assert got["capacities"].tolist() == want["capacities"], (
        f"capacities：{got['capacities'].tolist()} vs {want['capacities']}")
    assert isinstance(got["kept"], (list, tuple)) and len(got["kept"]) == len(caches), "kept 必须每层一项"
    for layer, row in enumerate(got["kept"]):
        length = caches[layer].shape[0]
        assert row.dtype == torch.int64 and row.dim() == 2 and row.shape[0] == caches[layer].shape[1], (
            f"第 {layer} 层 kept 必须是 (Hkv, n) 的 int64，got {tuple(row.shape)}")
        assert row.tolist() == want["kept"][layer], (
            f"第 {layer} 层 kept：{row.tolist()} vs {want['kept'][layer]}")
        for one in row.tolist():
            assert one == sorted(one) and len(set(one)) == len(one), f"kept 每行必须升序去重：{one}"
            assert one[-window:] == list(range(length - window, length)), f"尾部 window 必须原样保留：{one}"
    if want["tsp_indices"] is None:
        assert (got["tsp_indices"], got["hidden_out"], got["position_out"]) == (None, None, None), (
            "不做传播时 tsp_indices/hidden_out/position_out 都必须是 None")
    else:
        assert got["tsp_indices"].dtype == torch.int64, "tsp_indices 必须是 int64"
        assert got["tsp_indices"].tolist() == want["tsp_indices"], (
            f"tsp_indices：{got['tsp_indices'].tolist()} vs {want['tsp_indices']}")
        assert got["hidden_out"].dtype == caches[0].dtype, "hidden_out 必须保持输入的 dtype"
        assert got["hidden_out"].shape == (len(want["tsp_indices"]), hidden.shape[1]), "hidden_out 形状不对"
        assert got["hidden_out"].tolist() == want["hidden_out"], "hidden_out 必须按 tsp_indices 逐行取"
        assert got["position_out"].dtype == torch.int64, "position_out 必须是 int64"
        assert got["position_out"].tolist() == want["position_out"], "position_out 必须是原始位置，不能重编号"
    return got


def check_example(solve):
    """题面示例：q_len=12、L=3、tsp_idx=1、window=2、retain=0.5、tsp_rate=0.25。"""
    c0 = torch.tensor([[0.1], [0.2], [0.9], [0.3], [0.4], [0.8], [0.5], [0.7], [0.6], [0.85],
                       [0.0], [0.0]], dtype=torch.float64)
    c1 = torch.tensor([[0.1], [0.5], [0.2], [0.3], [0.95], [0.4], [0.6], [0.2], [0.7], [0.1],
                       [0.0], [0.0]], dtype=torch.float64)
    c2 = torch.tensor([[0.3], [0.2], [0.1]], dtype=torch.float64)
    hidden = torch.stack([torch.arange(12), torch.arange(100, 112)], 1).to(torch.float64)
    got = _verify(solve, [c0, c1, c2], 12, 1, hidden, torch.arange(100, 112), 2, 0.5, 0.25)
    assert got["capacities"].tolist() == [6, 6, 6], got["capacities"].tolist()
    assert got["kept"][0].tolist() == [[2, 5, 7, 9, 10, 11]], got["kept"][0].tolist()
    assert got["kept"][1].tolist() == [[1, 4, 6, 8, 10, 11]], got["kept"][1].tolist()
    assert got["kept"][2].tolist() == [[0, 1, 2]], "第 2 层容量 6 大于缓存 3，必须原样保留"
    assert got["tsp_indices"].tolist() == [4, 10, 11], got["tsp_indices"].tolist()
    assert got["hidden_out"].tolist() == [[4.0, 104.0], [10.0, 110.0], [11.0, 111.0]]
    assert got["position_out"].tolist() == [104, 110, 111], "位置必须保留原始绝对下标"


check_example.example = True


def check_capacity_widens_after_tsp_layer(solve):
    """retain_rate/tsp_rate 只作用于 TSP 层之后的层：缓存变短，预算按比例放大。"""
    case = _case(5, 40, 1, 0, 3, 2, 0.1, 0.25)
    got = _verify(solve, *case)
    assert [c.shape[0] for c in case[0]] == [40, 10, 10], "后续层的 cache 只有 tsp_length 个 token"
    assert got["capacities"].tolist() == [4, 4, 4], got["capacities"].tolist()
    assert got["kept"][2].shape == (1, 4), "第 2 层仍要压到 4 个 token，而不是按 0.1 压到 1 个"


def check_compression_is_per_head(solve):
    """逐 head 选 token：两个 head 的分数排序不同，kept 必须分 head 给出。"""
    case = _case(11, 16, 2, 0, 2, 2, 0.5, 0.25)
    case[0][0][:, 1] = case[0][0][:, 1].flip(0)
    got = _verify(solve, *case)
    rows = got["kept"][0].tolist()
    assert len(rows) == 2 and rows[0] != rows[1], f"两个 head 的保留集合必须不同：{rows}"
    assert got["tsp_indices"].shape == (4,), "tsp_indices 跨 head 只有一份"


def check_tsp_scores_are_summed_over_heads(solve):
    """TSP 用跨 head 求和后的分数：行 2 与行 5 求和后同为 1.0，同分取更小下标。"""
    past = torch.zeros(10, 2, dtype=torch.float64)
    past[2, 0], past[5, 0], past[2, 1], past[5, 1] = 0.9, 0.1, 0.1, 0.9
    c0 = torch.cat([past, torch.zeros(2, 2, dtype=torch.float64)])
    hidden = torch.stack([torch.arange(12), torch.arange(50, 62)], 1).to(torch.float64)
    got = _verify(solve, [c0], 12, 0, hidden, torch.arange(12), 2, 0.5, 0.3)
    assert got["tsp_indices"].tolist() == [2, 10, 11], got["tsp_indices"].tolist()
    assert got["kept"][0].tolist() == [[0, 1, 2, 5, 10, 11]] * 2, (
        "逐 head 各挑 4 个：head0 拿 2/5，head1 拿 5/2，升序后同集合")


def check_ties_keep_the_smaller_index(solve):
    """全同分：past 按更小下标补齐，window 恒在尾部。"""
    case = _case(3, 20, 1, 0, 1, 2, 0.5, 0.25)
    case[0] = [torch.ones_like(c) for c in case[0]]
    got = _verify(solve, *case)
    assert got["kept"][0].tolist() == [[0, 1, 2, 3, 4, 5, 6, 7, 18, 19]], got["kept"][0].tolist()
    assert got["tsp_indices"].tolist() == [0, 1, 2, 18, 19], got["tsp_indices"].tolist()


def check_no_propagation_when_tsp_rate_is_one(solve):
    """tsp_rate=1.0 时 tsp_length == q_len，`q_len > tsp_length` 不成立，不传播。"""
    case = _case(7, 12, 1, 1, 3, 2, 0.5, 1.0)
    got = _verify(solve, *case)
    assert got["tsp_indices"] is None and got["hidden_out"] is None and got["position_out"] is None
    assert got["capacities"].tolist() == [6, 6, 6], got["capacities"].tolist()


def check_random_against_oracle(solve):
    """随机形状/层数/比例，含 float32；容量、kept、TSP 下标与位置逐项对拍。"""
    cases = [_case(21, 20, 1, 0, 3, 3, 0.5, 0.25), _case(22, 16, 2, 1, 4, 2, 0.4, 0.2),
             _case(23, 24, 3, 0, 2, 4, 0.3, 0.25), _case(24, 13, 1, 2, 5, 1, 0.6, 0.3),
             _case(25, 40, 2, 1, 3, 4, 0.2, 0.25)]
    for case in cases:
        _verify(solve, *case)
    _verify(solve, *_case(26, 32, 2, 0, 2, 4, 0.5, 0.5, dtype=torch.float32))


def check_value_errors(solve):
    """非法输入（形状、dtype、比例、预算）必须抛 ValueError。"""
    caches, q_len, tsp_idx, hidden, pos, window, retain, tsp_rate = _case(1, 12, 1, 1, 3, 2, 0.5, 0.25)
    bad = [([], q_len, tsp_idx, hidden, pos, window, retain, tsp_rate),
           ([caches[0].flatten()], q_len, 0, hidden, pos, window, retain, tsp_rate),
           ([caches[0], caches[1].float()], q_len, tsp_idx, hidden, pos, window, retain, tsp_rate),
           ([caches[0], torch.zeros(3, 2, dtype=torch.float64)], q_len, tsp_idx, hidden, pos, window, retain, tsp_rate),
           (caches, 12.0, tsp_idx, hidden, pos, window, retain, tsp_rate),
           (caches, q_len, 3, hidden, pos, window, retain, tsp_rate),
           (caches, q_len, tsp_idx, hidden, pos, 0, retain, tsp_rate),
           (caches, q_len, tsp_idx, hidden, pos, window, 0.0, tsp_rate),
           (caches, q_len, tsp_idx, hidden, pos, window, retain, 1.5),
           (caches, q_len, tsp_idx, hidden, pos, window, retain, 0.1),
           (caches, q_len, tsp_idx, hidden.float(), pos, window, retain, tsp_rate),
           (caches, q_len, tsp_idx, hidden[:10], pos, window, retain, tsp_rate),
           (caches, q_len, tsp_idx, hidden, pos.float(), window, retain, tsp_rate),
           (caches, q_len, tsp_idx, hidden, pos[:11], window, retain, tsp_rate),
           ([caches[0], caches[1], torch.zeros(5, 1, dtype=torch.float64)], q_len, tsp_idx, hidden, pos, window, retain, tsp_rate),
           (caches, q_len, tsp_idx, hidden, pos, window, 0.05, tsp_rate)]
    for i, args in enumerate(bad):
        try:
            solve(*args)
        except ValueError:
            continue
        raise AssertionError(f"第 {i} 个非法输入应抛 ValueError")
