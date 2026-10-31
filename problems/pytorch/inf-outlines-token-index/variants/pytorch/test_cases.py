from collections import deque


def _oracle(transitions, match_states, sink_state, vocab, eos_token_id):
    """独立重写的 oracle：deque 两端操作实现 LIFO、预计算「有无出边」
    字典、bytes 逐步扫描用显式索引循环——与 solution 不同代码路径，同官方
    outlines-core index.rs 语义（token 插入序、match 即 final、eos 自环、
    首个无有效转移的非 match state 报错）。"""
    if not isinstance(transitions, dict) or not isinstance(vocab, dict):
        raise ValueError("bad input types")

    # 预计算：每个 state 是否存在不落 sink 的出边（useful 判定第二分支）
    has_edge = {}
    for s, row in transitions.items():
        has_edge[s] = any(v != sink_state for v in row.values())

    def step(st, byte):
        return transitions.get(st, {}).get(byte, sink_state)

    finals = set()
    edges = {}
    pending = deque([0])
    visited = {0}
    cache = {}

    while pending:
        cur = pending.popleft()  # appendleft + popleft == LIFO 栈
        valid = False
        if cur in match_states:
            finals.add(cur)
            valid = True
        for tok, ids in vocab.items():
            if eos_token_id in ids:
                continue
            st = cur
            alive = True
            data = tok.encode("utf-8")
            for k in range(len(data)):
                st = step(st, data[k])
                if st == sink_state:
                    alive = False
                    break
            if not alive:
                continue
            if st not in cache:
                cache[st] = (st in match_states) or has_edge.get(st, False)
            if cache[st]:
                valid = True
                row = edges.setdefault(cur, {})
                for tid in ids:
                    row[tid] = st
                if st not in visited:
                    visited.add(st)
                    pending.appendleft(st)
        if not valid and cur not in match_states:
            missing = []
            for b in range(256):
                if step(cur, b) != sink_state:
                    missing.append(chr(b) if b < 128 else "\\x%02x" % b)
            return {
                "incompatible": {"state": cur, "missing_tokens": missing}
            }

    for f in finals:
        edges.setdefault(f, {})[eos_token_id] = f
    return {
        "initial_state": 0,
        "final_states": sorted(finals),
        "transitions": edges,
        "incompatible": None,
    }


def _compare(solve, transitions, match_states, sink_state, vocab, eos):
    exp = _oracle(transitions, match_states, sink_state, vocab, eos)
    got = solve(transitions, match_states, sink_state, vocab, eos)
    assert isinstance(got, dict), f"返回应为 dict，得到 {type(got)}"
    if exp["incompatible"] is not None:
        assert got.get("incompatible") == exp["incompatible"], (
            f"incompatible 不一致：期望 {exp['incompatible']}，得到 {got.get('incompatible')}"
        )
    else:
        assert got.get("incompatible") is None, f"不应报错，得到 {got.get('incompatible')}"
        assert got.get("initial_state") == 0, f"initial_state 应为 0，得到 {got.get('initial_state')}"
        assert got.get("final_states") == exp["final_states"], (
            f"final_states 不一致：期望 {exp['final_states']}，得到 {got.get('final_states')}"
        )
        assert got.get("transitions") == exp["transitions"], (
            f"transitions 不一致：\n期望 {exp['transitions']}\n得到 {got.get('transitions')}"
        )
    return got


def _int_row(pairs):
    """{byte: next} 的便捷构造：键用 int 字节。"""
    return dict(pairs)


def check_basic_int_dfa(solve):
    # regex `0|[1-9][0-9]*` 的 DFA（官方测试同款词表）：
    # 0 --'0'--> 1(match)；0 --[1-9]--> 2(match)；2 --[0-9]--> 2 自环
    t = {
        0: _int_row([(0x30, 1)] + [(0x31 + i, 2) for i in range(9)]),
        2: _int_row([(0x30 + i, 2) for i in range(10)]),
    }
    vocab = {"blah": [0], "1a": [1], "2": [2], "0": [3]}
    led = _compare(solve, t, {1, 2}, 3, vocab, 4)
    # pop 0：blah b→sink 丢；1a: '1'→2 后 'a'→sink 整体作废；
    # "2"→2（match，有用）加边 {2:2}；"0"→1（match）加边 {3:1}
    assert led["transitions"][0] == {2: 2, 3: 1}
    # state 1 是 match 但无 token 出边 → 不报错，仅 eos 自环
    assert led["transitions"][1] == {4: 1}
    # state 2：blah/1a 作废；"2" 自环、"0" 自环（'0' 从 2 出发仍到 2）
    assert led["transitions"][2] == {2: 2, 3: 2, 4: 2}
    assert led["final_states"] == [1, 2]
    # match 非吸收：从 final 出发的普通 token 边仍然保留
    assert 2 in led["transitions"][2] and 4 in led["transitions"][2]


def check_initial_match(solve):
    # state 0 即 match（可空模式 a*）：final 含 0，eos 自环与 token 边共存
    t = {0: _int_row([(ord("a"), 0)])}
    led = _compare(solve, t, {0}, 1, {"a": [7]}, 9)
    assert led["final_states"] == [0]
    assert led["transitions"] == {0: {7: 0, 9: 0}}


def check_over_match_continue(solve):
    # regex `a+`：越过 match state 继续推进（match 不是吸收态）
    t = {0: _int_row([(ord("a"), 1)]), 1: _int_row([(ord("a"), 1)])}
    led = _compare(solve, t, {1}, 2, {"a": [0], "aa": [1]}, 3)
    assert led["final_states"] == [1]
    assert led["transitions"][0] == {0: 1, 1: 1}
    assert led["transitions"][1] == {0: 1, 1: 1, 3: 1}


def check_incompatible_basic(solve):
    # regex `ab` 但词表没有能从 state 1 出发的 token：
    # 1 非 match 且无有效 token 边 → IncompatibleVocabulary，缺 'b'
    t = {0: _int_row([(ord("a"), 1)]), 1: _int_row([(ord("b"), 2)])}
    led = _compare(solve, t, {2}, 3, {"a": [0], "ax": [1]}, 4)
    assert led["incompatible"] == {"state": 1, "missing_tokens": ["b"]}


def check_incompatible_hex_and_order(solve):
    # 缺失字节为非 ascii 时输出 \xHH（两位小写十六进制）；多字节按 byte 升序
    t = {0: _int_row([(0x41, 1), (0xF0, 2)])}
    led = _compare(solve, t, set(), 3, {"x": [0]}, 9)
    assert led["incompatible"] == {"state": 0, "missing_tokens": ["A", "\\xf0"]}


def check_incompatible_dfs_order(solve):
    # 两个 state 都不兼容：LIFO 枚举序决定先报谁。
    # 0 --a--> 1 --c--> 2(match)；0 --b--> 3 --d--> 4(match)
    # pop 0 后栈 [1,3]，LIFO 先处理 3 → 先报 3（FIFO 会先报 1）
    t = {
        0: _int_row([(ord("a"), 1), (ord("b"), 3)]),
        1: _int_row([(ord("c"), 2)]),
        3: _int_row([(ord("d"), 4)]),
    }
    led = _compare(solve, t, {2, 4}, 5, {"a": [0], "b": [1], "zz": [2]}, 6)
    assert led["incompatible"] == {"state": 3, "missing_tokens": ["d"]}


def check_dead_end_not_error(solve):
    # 死胡同 state（非 match 且无任何 DFA 出边）永不入队 → 不报错、
    # 不出现在 index；到达它的 token 整体作废，但出发 state 因其他
    # token 有效而不报错。
    # 0 --a--> 1（死胡同）；0 --b--> 2(match self)
    t = {0: _int_row([(ord("a"), 1), (ord("b"), 2)]), 2: _int_row([(ord("b"), 2)])}
    led = _compare(solve, t, {2}, 3, {"a": [0], "b": [1]}, 4)
    assert led["incompatible"] is None
    assert led["final_states"] == [2]
    assert led["transitions"] == {0: {1: 2}, 2: {1: 2, 4: 2}}
    # state 1 完全不出现
    assert 1 not in led["transitions"]
    assert led["transitions"][0] == {1: 2}


def check_unreached_match_not_final(solve):
    # 从未被 token 到达、也不在 state 0 的 match state 不进 final_states。
    # 0 --a--> 1 --b--> 2(match)；孤立 3 也是 match 但不可达
    t = {0: _int_row([(ord("a"), 1)]), 1: _int_row([(ord("b"), 2)])}
    led = _compare(solve, t, {2, 3}, 4, {"a": [0], "ab": [1], "b": [5]}, 6)
    assert led["final_states"] == [2]
    assert 3 not in led["transitions"]
    assert led["transitions"][0] == {0: 1, 1: 2}
    assert led["transitions"][1] == {5: 2}
    assert led["transitions"][2] == {6: 2}


def check_stop_at_intermediate(solve):
    # token 可以停在非 match 的中段 state（该 state 有 DFA 出边 → 有用），
    # 边照加、state 照入队 —— "是否 match" 不影响收录，只影响 final 判定
    # DFA: 0 --a--> 1 --b--> 2(match)；vocab 恰好覆盖两段
    t = {0: _int_row([(ord("a"), 1)]), 1: _int_row([(ord("b"), 2)])}
    led = _compare(solve, t, {2}, 3, {"a": [0], "b": [1]}, 4)
    assert led["final_states"] == [2]
    assert led["transitions"][0] == {0: 1}
    assert led["transitions"][1] == {1: 2}
    assert led["transitions"][2] == {4: 2}


def check_eos_token_skipped(solve):
    # 含 eos id 的 token 整个跳过（即使它本身可走）；eos 只出现在 final 自环
    # regex `[ab]+`：0 --a/b--> 1(match self)
    t = {
        0: _int_row([(ord("a"), 1), (ord("b"), 1)]),
        1: _int_row([(ord("a"), 1), (ord("b"), 1)]),
    }
    led = _compare(solve, t, {1}, 2, {"a": [0, 3], "b": [1], "c": [2]}, 3)
    assert led["final_states"] == [1]
    assert led["transitions"][0] == {1: 1}
    assert led["transitions"][1] == {1: 1, 3: 1}
    # eos id 3 绝不作为普通 token 边出现
    for row in led["transitions"].values():
        assert set(row) <= {1, 3}


def check_multi_id_edge(solve):
    # 一词多 id：同一 token 的全部 id 各自建一条边指向同一 next state
    t = {0: _int_row([(ord("a"), 1)]), 1: _int_row([(ord("a"), 1)])}
    led = _compare(solve, t, {1}, 2, {"aa": [5, 6]}, 7)
    assert led["final_states"] == [1]
    assert led["transitions"][0] == {5: 1, 6: 1}
    assert led["transitions"][1] == {5: 1, 6: 1, 7: 1}


def check_multibyte_token_walk(solve):
    # 多字节 token 按 UTF-8 字节逐步推进：任何字节落 sink 即整体作废，
    # 即使此前已经路过合法前缀。注意 "\xf0" 在 str 里是字符 ð（c3 b0）。
    # DFA 接受 😀 的完整四字节 f0 9f 98 80：
    # 0 --f0--> 1 --9f--> 2 --98--> 3 --80--> 4(match)
    t = {
        0: _int_row([(0xF0, 1)]),
        1: _int_row([(0x9F, 2)]),
        2: _int_row([(0x98, 3)]),
        3: _int_row([(0x80, 4)]),
    }
    vocab = {"\U0001f600": [0], "\xf0": [1]}  # 😀 四字节；ð = c3 b0 从 0 出发即死
    led = _compare(solve, t, {4}, 5, vocab, 6)
    # pop 0：😀 走满四字节到 4（match 有用）加边 {0:4} 入栈；ð 的 c3 落 sink 作废
    assert led["transitions"][0] == {0: 4}
    assert led["final_states"] == [4]
    # 中段 1/2/3 非 match 且无出边之外可走？1/2/3 各有 DFA 出边（useful），
    # 但无 token 能从它们出发（首字节 f0/9f/98 不在任何 token 的后续里）
    # —— 不对：它们根本没被任何完整 token 到达（😀 跳过中段直达 4），
    # 死在半路的 token 已作废，中段 state 不入队、不报错、不收录
    assert led["transitions"][4] == {6: 4}
    assert led["incompatible"] is None
