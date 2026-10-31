from itertools import product


def _py(probs, beam_width=3, blank_id=0, k=None):
    """oracle：独立实现——穷举全部 V^T 路径，按 collapse 求和每条输出序列的概率。"""
    T = len(probs)
    V = len(probs[0])
    acc = {}
    for path in product(range(V), repeat=T):
        p = 1.0
        for t, c in enumerate(path):
            p *= probs[t][c]
        out = []
        prev = None
        for c in path:
            if c != prev and c != blank_id:
                out.append(c)
            prev = c
        key = tuple(out)
        acc[key] = acc.get(key, 0.0) + p
    items = sorted(acc.items(), key=lambda kv: (-kv[1], kv[0]))
    kk = beam_width if k is None else k
    return {"hypotheses": [{"prefix": list(s), "prob": pr}
                           for s, pr in items[:kk] if pr > 0.0]}


def _compare(solve, probs, beam_width=3, blank_id=0, k=None):
    led = solve(probs, beam_width, blank_id, k)
    ref = _py(probs, beam_width, blank_id, k)
    assert led == ref, f"solve vs oracle:\n{led}\nvs\n{ref}"
    return led


def check_anchor_single_frame(solve):
    led = _compare(solve, [[0.2, 0.5, 0.3]], 3, 0)
    assert led["hypotheses"] == [
        {"prefix": [1], "prob": 0.5},
        {"prefix": [2], "prob": 0.3},
        {"prefix": [], "prob": 0.2},
    ]


def check_repeat_and_blank_split(solve):
    # probs: 帧1 是 blank 0.9。collapse 到 [1] 的路径概率和 = 0.262（穷举核对）
    probs = [[0.1, 0.9], [0.9, 0.1], [0.1, 0.9]]
    led = _compare(solve, probs, 4, 0)
    hyp = {tuple(h["prefix"]): h["prob"] for h in led["hypotheses"]}
    assert abs(hyp[(1,)] - 0.262) < 1e-12
    assert abs(hyp[()] - 0.009) < 1e-12
    assert abs(hyp[(1, 1)] - 0.729) < 1e-12  # A blank A
    # AA 重复：延续路径并入同前缀
    probs2 = [[0.5, 0.5], [0.5, 0.5]]
    led2 = _compare(solve, probs2, 4, 0)
    hyp2 = {tuple(h["prefix"]): h["prob"] for h in led2["hypotheses"]}
    assert abs(hyp2[(1,)] - 0.75) < 1e-12  # P(至少一个 A) = 0.75
    assert abs(hyp2[()] - 0.25) < 1e-12


def check_beam_pruning_changes_result(solve):
    probs = [[0.3, 0.4, 0.3], [0.6, 0.2, 0.2]]
    # beam1：帧0 只保留 (1,)，其 blank 概率计入 (1,) 自身（() 不在 beam）
    led1 = solve(probs, 1, 0)
    assert led1["hypotheses"] == [{"prefix": [1], "prob": 0.32}]
    # beam3：与穷举一致，top1 (1,) 0.38（blank 后前缀仍是 (1,)，不回 ()）
    led3 = solve(probs, 3, 0)
    assert led3["hypotheses"][0] == {"prefix": [1], "prob": 0.38}


def check_tie_lexicographic(solve):
    led = _compare(solve, [[0.5, 0.5]], 2, 0)
    assert [tuple(h["prefix"]) for h in led["hypotheses"]] == [(), (1,)]
    led2 = _compare(solve, [[0.25, 0.5, 0.25]], 3, 0)
    assert [tuple(h["prefix"]) for h in led2["hypotheses"]] == [(1,), (), (2,)]


def check_k_truncation(solve):
    probs = [[0.1, 0.6, 0.3], [0.1, 0.6, 0.3]]
    led_k1 = _compare(solve, probs, 3, 0, 1)
    led_all = _compare(solve, probs, 3, 0, None)
    assert led_k1["hypotheses"] == led_all["hypotheses"][:1]
    assert len(led_all["hypotheses"]) <= 3


def check_prob_zero_and_degenerate(solve):
    # 全 blank 概率 → 空输出
    led = _compare(solve, [[1.0, 0.0], [1.0, 0.0]], 3, 0)
    assert led["hypotheses"][0]["prefix"] == []
    assert abs(led["hypotheses"][0]["prob"] - 1.0) < 1e-12
    # 概率含 0：路径剪除不影响账
    led2 = _compare(solve, [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], 3, 0)
    # A blank A → collapse 输出 [1, 1]（两段）
    assert led2["hypotheses"][0]["prefix"] == [1, 1]
    assert abs(led2["hypotheses"][0]["prob"] - 1.0) < 1e-12


def check_invalid_inputs(solve):
    def ve(*args):
        try:
            solve(*args)
        except ValueError:
            return
        raise AssertionError("expected ValueError")
    ve([], 3, 0)
    ve([[]], 3, 0)
    ve([[0.5, 0.5], [0.5]], 3, 0)          # 行长不齐
    ve([[-0.1, 0.5]], 3, 0)                 # 负概率
    ve([[0.5, float("nan")]], 3, 0)
    ve([[0.5, True]], 3, 0)                 # bool
    ve([[0.5, 0.5]], 3, 2)                  # blank 越界
    ve([[0.5, 0.5]], 3, -1)
    ve([[0.5, 0.5]], 0, 0)                  # beam_width 0
    ve([[0.5, 0.5]], True, 0)               # beam_width bool
    ve([[0.5, 0.5]], 3, 0, 0)               # k 0
    ve([[0.5, 0.5]], 3, 0, True)            # k bool
    ve("nope", 3, 0)


def check_oracle_random(solve):
    import random
    rng = random.Random(20260670)
    for _ in range(120):
        T = rng.randint(1, 4)
        V = rng.randint(2, 3)
        blank = rng.randint(0, V - 1)
        # 随机 logits → softmax 归一为合法概率
        import math
        probs = []
        for _t in range(T):
            logits = [rng.uniform(-2, 2) for _ in range(V)]
            if rng.random() < 0.5:
                logits[rng.randrange(V)] += rng.uniform(2, 5)  # 造尖峰
            m = max(logits)
            exps = [math.exp(x - m) for x in logits]
            s = sum(exps)
            probs.append([e / s for e in exps])
        # 穷举 oracle 语义：bw=999 保证不剪枝（T≤4、V≤3 时前缀空间远小于 999）
        bw = 999
        led = solve(probs, bw, blank, None)
        ref = _py(probs, bw, blank, None)
        assert len(led["hypotheses"]) == len(ref["hypotheses"])
        for a, b in zip(led["hypotheses"], ref["hypotheses"]):
            assert a["prefix"] == b["prefix"]
            assert abs(a["prob"] - b["prob"]) < 1e-9
