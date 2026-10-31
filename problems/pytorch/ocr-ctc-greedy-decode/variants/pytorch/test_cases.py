def _py(logits, blank_id=0):
    """oracle：官方 selection 掩码实现路径（先相邻去重掩码、再叠加 blank 过滤）。"""
    V = len(logits[0])
    raw = []
    for row in logits:
        vals = list(row)
        best = vals.index(max(vals))  # max 相同时 index 取首个 → 最小 id
        raw.append(best)
    T = len(raw)
    selection = [True] * T
    for t in range(1, T):
        selection[t] = raw[t] != raw[t - 1]
    for t in range(T):
        if raw[t] == blank_id:
            selection[t] = False
    tokens = [raw[t] for t in range(T) if selection[t]]
    # spans：与 selection 等价的重建——重复帧（raw[t]==raw[t-1]）扩展当前 run，
    # blank 中断、不同 id 开新 run
    runs = []  # [start, end, id]
    for t in range(T):
        if raw[t] == blank_id:
            continue
        if runs and raw[t] == raw[t - 1]:
            runs[-1][1] = t
        else:
            runs.append([t, t, raw[t]])
    spans = [[s, e] for s, e, _ in runs]
    return {"raw_path": raw, "tokens": tokens, "spans": spans}


def _compare(solve, logits, blank_id=0):
    led = solve(logits, blank_id)
    ref = _py(logits, blank_id)
    assert led == ref, f"solve vs oracle:\n{led}\nvs\n{ref}"
    return led


def check_anchor_basic(solve):
    # path [1,1,0,2,2,2] blank=0 → tokens [1,2]，spans [[0,1],[3,5]]
    logits = [[0.1, 5.0, 0.1], [0.1, 4.0, 0.1], [5.0, 0.1, 0.1],
              [0.1, 0.1, 3.0], [0.1, 0.1, 3.0], [0.1, 0.1, 3.0]]
    led = _compare(solve, logits, 0)
    assert led["raw_path"] == [1, 1, 0, 2, 2, 2]
    assert led["tokens"] == [1, 2]
    assert led["spans"] == [[0, 1], [3, 5]]


def check_blank_split(solve):
    # A blank A：blank 在原始序列上隔开两个 A → 都保留（CTC 语义核心）
    logits = [[0.1, 5.0], [5.0, 0.1], [0.1, 5.0]]
    led = _compare(solve, logits, 0)
    assert led["tokens"] == [1, 1]
    assert led["spans"] == [[0, 0], [2, 2]]
    # A A blank A A → 前两个合并、后两个合并、blank 分隔
    logits2 = [[0.1, 5.0], [0.1, 5.0], [5.0, 0.1], [0.1, 5.0], [0.1, 5.0]]
    led2 = _compare(solve, logits2, 0)
    assert led2["tokens"] == [1, 1]
    assert led2["spans"] == [[0, 1], [3, 4]]


def check_all_blank(solve):
    logits = [[5.0, 0.1]] * 4
    led = _compare(solve, logits, 0)
    assert led["tokens"] == [] and led["spans"] == []
    assert led["raw_path"] == [0, 0, 0, 0]


def check_leading_trailing_blank(solve):
    logits = [[5.0, 0.1], [0.1, 5.0], [0.1, 5.0], [5.0, 0.1], [0.1, 5.0]]
    led = _compare(solve, logits, 0)
    assert led["tokens"] == [1, 1]
    assert led["spans"] == [[1, 2], [4, 4]]


def check_argmax_tie(solve):
    # 完全并列取最小 id；略大的值打破并列
    logits = [[1.0, 1.0, 1.0], [0.1, 0.2, 0.3]]
    led = _compare(solve, logits, 2)  # blank=2：帧0 并列 → id0，帧1 → id2(blank)
    assert led["raw_path"] == [0, 2]
    assert led["tokens"] == [0] and led["spans"] == [[0, 0]]
    logits2 = [[0.1, 0.1], [2.0, 2.0]]
    led2 = _compare(solve, logits2, 0)
    # 两帧并列 → id0 = blank → 输出为空
    assert led2["raw_path"] == [0, 0] and led2["tokens"] == [] and led2["spans"] == []


def check_single_frame_and_alternating(solve):
    # 单帧；交替 token 无相邻重复
    led = _compare(solve, [[0.1, 5.0]], 0)
    assert led["tokens"] == [1] and led["spans"] == [[0, 0]]
    logits = [[0.1, 5.0, 0.1], [5.0, 0.1, 0.1], [0.1, 0.1, 5.0],
              [5.0, 0.1, 0.1]]
    led2 = _compare(solve, logits, 0)
    # 帧1/帧3 的 argmax 是 id0 = blank，被丢弃
    assert led2["tokens"] == [1, 2]
    assert led2["spans"] == [[0, 0], [2, 2]]


def check_custom_blank(solve):
    # blank_id 非 0：id1 是 blank，相同账目换角色
    logits = [[5.0, 0.1, 0.1], [0.1, 0.1, 5.0], [0.1, 0.1, 5.0]]
    led = _compare(solve, logits, 1)
    assert led["tokens"] == [0, 2]
    assert led["spans"] == [[0, 0], [1, 2]]


def check_invalid_inputs(solve):
    def ve(*args):
        try:
            solve(*args)
        except ValueError:
            return
        raise AssertionError("expected ValueError")
    ve([], 0)
    ve([[]], 0)
    ve([[1.0, 2.0], [1.0]], 0)          # 行长不齐
    ve([[1.0, "x"]], 0)                  # 非数
    ve([[1.0, 2.0]], True)               # blank bool
    ve([[1.0, 2.0]], 2)                  # blank 越界
    ve([[1.0, 2.0]], -1)
    ve([[float("nan"), 2.0]], 0)
    ve([[float("inf"), 2.0]], 0)
    ve("nope", 0)


def check_oracle_random(solve):
    import random
    rng = random.Random(20260668)
    for _ in range(200):
        T = rng.randint(1, 20)
        V = rng.randint(2, 6)
        blank = rng.randint(0, V - 1)
        logits = []
        for _t in range(T):
            row = [rng.uniform(-3, 3) for _ in range(V)]
            if rng.random() < 0.6:  # 大概率造尖峰，路径多重复/blank
                c = rng.randint(0, V - 1)
                row[c] += rng.uniform(4, 9)
            logits.append(row)
        led = solve(logits, blank)
        ref = _py(logits, blank)
        assert led == ref, "random mismatch"
