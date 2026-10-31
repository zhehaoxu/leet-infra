import torch


KEYS = {"prefill_log_index", "prefill_local_index", "final_log_index",
        "final_local_index", "final_fp_positions"}


def _even_positions(seq, count):
    """seq[0:count:2]，独立按「下标为偶数且 < count」重算。"""
    return [seq[i] for i in range(0, min(count, len(seq)), 2)]


def _tail(seq, count):
    return list(seq[count:])


def _step(log, local, seen, W):
    seen += 1
    local = local + [seen - 1]
    if len(local) > 2 * W:
        head = local[:W]
        log = _even_positions(log + head, len(log) + W) if log else head
        local = _tail(local, W)
    return log, local, seen


def _oracle(L, T, W):
    seen, log, local = 0, [], []
    if L > 1:
        seen = L
        local = list(range(L))
        while True:
            tmp = log + local
            if len(tmp) < 3 * W:
                break
            log = _even_positions(tmp, 2 * W)
            local = _tail(tmp, 2 * W)
        if W < L < 3 * W:
            tmp = log + local
            log, local = list(tmp[:W]), _tail(tmp, W)
    elif L == 1:
        log, local, seen = _step(log, local, seen, W)
    prefill = (list(log), list(local))
    for _ in range(T):
        log, local, seen = _step(log, local, seen, W)
    return {
        "prefill_log_index": prefill[0],
        "prefill_local_index": prefill[1],
        "final_log_index": log,
        "final_local_index": local,
        "final_fp_positions": sorted(set(log) | set(local)),
    }


def _verify(solve, L, T, W):
    got = solve(L, T, W)
    assert isinstance(got, dict) and set(got) == KEYS, f"键不对：{sorted(got)}"
    exp = _oracle(L, T, W)
    for k, v in exp.items():
        t = got[k]
        assert torch.is_tensor(t) and t.dtype == torch.int64, f"{k} 不是 int64：{t.dtype}"
        assert t.dim() == 1, f"{k} 不是一维：{tuple(t.shape)}"
        assert t.tolist() == v, f"solve({L}, {T}, {W}) 的 {k} = {t.tolist()}，期望 {v}"
    return got


def check_example(solve):
    """题面三个例子：三轮折叠、解码触发合并、覆盖分支与 L == W 空 log。"""
    r = _verify(solve, 20, 0, 4)
    assert r["prefill_log_index"].tolist() == [0, 8, 12, 14]
    assert r["prefill_local_index"].tolist() == [16, 17, 18, 19]
    assert r["final_fp_positions"].tolist() == [0, 8, 12, 14, 16, 17, 18, 19]

    r = _verify(solve, 12, 5, 4)
    assert r["prefill_log_index"].tolist() == [0, 2, 4, 6]
    assert r["final_log_index"].tolist() == [0, 4, 8, 10]
    assert r["final_local_index"].tolist() == [12, 13, 14, 15, 16]

    assert _verify(solve, 7, 0, 4)["final_log_index"].tolist() == [0, 1, 2, 3]
    assert _verify(solve, 4, 0, 4)["final_log_index"].numel() == 0


check_example.example = True


def check_fold_is_even_positions_of_the_concat(solve):
    """折叠取「整段前 2W 个的偶数位」，不是各抽一半：L=16 折两轮，头部只留 {0,8}。"""
    r = _verify(solve, 16, 0, 4)
    assert r["prefill_log_index"].tolist() == [0, 4, 8, 10], r["prefill_log_index"].tolist()
    assert r["prefill_local_index"].tolist() == [12, 13, 14, 15]
    r = _verify(solve, 24, 0, 4)
    assert r["prefill_log_index"].tolist() == [0, 12, 16, 18], r["prefill_log_index"].tolist()
    assert r["prefill_local_index"].tolist() == [20, 21, 22, 23]


def check_threshold_is_ge_three_w(solve):
    """L = 3W 恰好触发一次折叠；L = 3W - 1 落进覆盖分支。"""
    assert _verify(solve, 12, 0, 4)["prefill_log_index"].tolist() == [0, 2, 4, 6]
    assert _verify(solve, 11, 0, 4)["prefill_log_index"].tolist() == [0, 1, 2, 3]
    assert _verify(solve, 11, 0, 4)["prefill_local_index"].tolist() == [4, 5, 6, 7, 8, 9, 10]


def check_override_keys_off_original_len(solve):
    """覆盖判据用原始 L，不看折叠后的 len(tmp)：L >= 3W 时即便只剩 2W 也不覆盖。"""
    assert _verify(solve, 12, 0, 4)["prefill_local_index"].tolist() == [8, 9, 10, 11]
    assert _verify(solve, 8, 0, 4)["prefill_log_index"].tolist() == [0, 1, 2, 3]


def check_len_equals_window_keeps_log_empty(solve):
    """L == W 既不折叠也不覆盖；L < W 同样空 log，位置全在滑窗里。"""
    for W in (1, 3, 4, 8):
        assert _verify(solve, W, 0, W)["prefill_log_index"].tolist() == []
        assert _verify(solve, W, 0, W)["prefill_local_index"].tolist() == list(range(W))
        assert _verify(solve, 1, 0, W)["final_local_index"].tolist() == [0]


def check_decode_needs_two_w_plus_one(solve):
    """解码阈值是 len(local) > 2W：第 2W 个不动，第 2W+1 个才并；空 log 的首次整段搬过去。"""
    r = _verify(solve, 0, 8, 4)
    assert r["final_log_index"].numel() == 0, r["final_log_index"].tolist()
    assert r["final_local_index"].tolist() == list(range(8))
    r = _verify(solve, 0, 9, 4)
    assert r["final_log_index"].tolist() == [0, 1, 2, 3]
    assert r["final_local_index"].tolist() == [4, 5, 6, 7, 8]
    r = _verify(solve, 0, 13, 4)
    assert r["final_log_index"].tolist() == [0, 2, 4, 6]
    assert r["final_local_index"].tolist() == [8, 9, 10, 11, 12]
    r = _verify(solve, 0, 18, 4)
    assert r["final_log_index"].tolist() == [0, 4, 8, 10]
    assert r["final_local_index"].tolist() == [12, 13, 14, 15, 16, 17]


def check_fp_budget_stays_three_w(solve):
    """解码期全精度总量恒为 3W：并集长度、两段都按预算走。"""
    for W in (2, 3, 4, 5):
        total = W + 3 * W
        r = _verify(solve, 0, total, W)
        n1 = r["final_log_index"].numel()
        n2 = r["final_local_index"].numel()
        assert n1 == W and n2 == 2 * W, (W, n1, n2)
        assert r["final_fp_positions"].numel() == 3 * W


def check_prefill_snapshot_is_before_decode(solve):
    """L == 1 与 L == 0 加一步解码等价；prefill_* 是进入解码步之前的快照。"""
    a = _verify(solve, 1, 6, 4)
    b = _verify(solve, 0, 7, 4)
    assert a["final_log_index"].tolist() == b["final_log_index"].tolist()
    assert a["final_local_index"].tolist() == b["final_local_index"].tolist()
    assert a["prefill_local_index"].tolist() == [0]
    assert b["prefill_log_index"].numel() == 0 and b["prefill_local_index"].numel() == 0
    c = _verify(solve, 20, 0, 4)
    assert c["final_log_index"].tolist() == c["prefill_log_index"].tolist()


def check_lists_are_disjoint_and_ascending(solve):
    """两段语义不相交、并集去重升序，且都是绝对下标。"""
    for L, T, W in [(20, 0, 4), (12, 5, 4), (37, 25, 6), (3, 40, 3), (1, 25, 4), (64, 30, 7)]:
        r = _verify(solve, L, T, W)
        log = r["final_log_index"].tolist()
        local = r["final_local_index"].tolist()
        assert log == sorted(log) and local == sorted(local)
        assert not (set(log) & set(local)), (L, T, W)
        assert r["final_fp_positions"].tolist() == sorted(set(log) | set(local))
        assert all(0 <= p < L + T for p in r["final_fp_positions"].tolist())
        assert local and local[-1] == L + T - 1, (L, T, W, local)


def check_against_oracle(solve):
    """网格遍历：折叠轮数、覆盖分支、解码合并次数全部与独立 oracle 对拍。"""
    for W in (1, 2, 3, 4, 5, 7):
        for L in (0, 1, 2, 3, 5, 8, 13, 21, 34):
            for T in (0, 1, 4, 11, 23):
                if L == 0 and T == 0:
                    continue                    # 两者同时为 0 是非法参数，见 check_value_errors
                _verify(solve, L, T, W)


def check_value_errors(solve):
    """类型、负数、W < 1、两个长度同时为 0 全部抛 ValueError；0 与正数另一侧合法。"""
    for i, args in enumerate([(3, 0, 1), (0, 3, 1), (5, 5, 4), (1, 0, 1),
                              (0, 1, 1), (2, 0, 1)]):
        _verify(solve, *args)
    for i, args in enumerate([(0, 0, 0), (1, 0, 0), (-1, 1, 2), (1, -1, 2),
                              (True, 1, 2), (1, True, 2), (1.0, 1, 2), (1, 1.0, 2),
                              ("1", 1, 2), (0, 0, 2)]):
        try:
            solve(*args)
        except ValueError:
            continue
        raise AssertionError(f"第 {i} 组非法参数没抛 ValueError：{args}")
