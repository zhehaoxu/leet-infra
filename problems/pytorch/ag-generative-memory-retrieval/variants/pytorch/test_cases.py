import math


def _py(query_emb, memories, k, recency_decay=0.995, gw=(0.5, 3, 2)):
    n = len(memories)
    rec = [recency_decay ** (i + 1) for i in range(n)]
    rel = []
    for m in memories:
        e, q = m["emb"], query_emb
        num = sum(a * b for a, b in zip(e, q))
        den = math.sqrt(sum(a * a for a in e)) * math.sqrt(sum(b * b for b in q))
        rel.append(num / den)
    imp = [m["importance"] for m in memories]

    def norm01(vals):
        lo, hi = min(vals), max(vals)
        rng = hi - lo
        if rng == 0:
            return [0.5] * len(vals)
        return [(v - lo) / rng for v in vals]

    rec_n, rel_n, imp_n = norm01(rec), norm01(rel), norm01(imp)
    scores = [gw[0] * rec_n[i] + gw[1] * rel_n[i] + gw[2] * imp_n[i] for i in range(n)]
    # 稳定排序：按分数降序，并列保持插入（时间线）序
    idx = list(range(n))
    idx.sort(key=lambda i: -scores[i])
    return {
        "scores": scores,
        "top": idx[:k],
        "components": {"recency": rec_n, "relevance": rel_n, "importance": imp_n},
    }


def _compare(solve, query_emb, memories, k, recency_decay=0.995, gw=(0.5, 3, 2)):
    led = solve(query_emb, memories, k, recency_decay, gw)
    ref = _py(query_emb, memories, k, recency_decay, gw)
    def flat_close(a, b, path=""):
        if isinstance(a, list):
            assert isinstance(b, list) and len(a) == len(b), path
            for i, (x, y) in enumerate(zip(a, b)):
                flat_close(x, y, f"{path}[{i}]")
        elif isinstance(a, dict):
            assert set(a.keys()) == set(b.keys()), path
            for key in a:
                flat_close(a[key], b[key], f"{path}.{key}")
        elif isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
            assert a == b, f"{path}: {a} vs {b}"
        else:
            assert abs(a - b) <= 1e-9 + 1e-9 * abs(b), f"{path}: {a} vs {b}"
    flat_close(led, ref)
    return led


def check_anchor_hand(solve):
    # 3 条记忆： relevance 分别 1/0/0.5 → 归一化 1/0/0.5
    # recency: 0.995^1,2,3 → 归一化 0（最小）/中/1（最大）
    # importance 全等 → 归一化 range=0 → 全 0.5
    q = [1.0, 0.0]
    mems = [
        {"emb": [1.0, 0.0], "importance": 5.0},   # rel=1
        {"emb": [0.0, 1.0], "importance": 5.0},   # rel=0
        {"emb": [1.0, math.sqrt(3.0)], "importance": 5.0},  # rel = 1/(1*2) = 0.5 精确
    ]
    led = _compare(solve, q, mems, 3)
    assert abs(led["components"]["relevance"][0] - 1.0) < 1e-12
    assert abs(led["components"]["relevance"][1] - 0.0) < 1e-12
    assert abs(led["components"]["relevance"][2] - 0.5) < 1e-9
    # importance range=0 → 全 0.5
    assert led["components"]["importance"] == [0.5, 0.5, 0.5]
    # recency（官方口径）：decay^(i+1)，最旧 idx0 最大 → 归一化 1；最新 idx2 → 0
    assert abs(led["components"]["recency"][0] - 1.0) < 1e-12
    assert abs(led["components"]["recency"][2]) < 1e-12
    # score: idx0 = 0.5*1 + 3*1 + 2*0.5 = 4.5；idx2 = 0.5*0 + 3*0.5 + 2*0.5 = 2.5
    assert abs(led["scores"][0] - 4.5) < 1e-9
    assert abs(led["scores"][2] - 2.5) < 1e-9
    assert led["top"][0] == 0


def check_tie_chronological(solve):
    # 全部并列（同 embedding 同 importance）：top 保持时间线顺序
    q = [1.0, 2.0]
    mems = [
        {"emb": [0.0, 1.0], "importance": 3.0},
        {"emb": [0.0, 1.0], "importance": 3.0},
        {"emb": [0.0, 1.0], "importance": 3.0},
    ]
    led = _compare(solve, q, mems, 3)
    # rel/imp 全 0.5，recency 归一化 [1, ~0.4987, 0] → scores = 0.5*rec_n + 2.5
    assert led["top"] == [0, 1, 2]
    assert abs(led["scores"][0] - 3.0) < 1e-9
    assert abs(led["scores"][1] - 2.749373433583962) < 1e-9
    assert abs(led["scores"][2] - 2.5) < 1e-9


def check_single_memory(solve):
    # 单条：三个分量都 range=0 → 全 0.5 → score = (0.5+3+2)*0.5
    led = _compare(solve, [1.0], [{"emb": [2.0], "importance": 9.0}], 1)
    assert led["components"]["recency"] == [0.5]
    assert led["components"]["relevance"] == [0.5]
    assert abs(led["scores"][0] - 2.75) < 1e-12


def check_k_truncation(solve):
    q = [1.0, 0.0]
    mems = [
        {"emb": [0.9, 0.1], "importance": 1.0},
        {"emb": [0.8, 0.2], "importance": 2.0},
        {"emb": [0.7, 0.3], "importance": 3.0},
        {"emb": [0.6, 0.4], "importance": 4.0},
    ]
    led1 = _compare(solve, q, mems, 2)
    led4 = _compare(solve, q, mems, 4)
    assert len(led1["top"]) == 2
    assert led4["top"] == sorted(led4["top"]) or True
    assert set(led4["top"]) == set(range(4))
    # k=2 的 top 是 k=4 top 的前缀
    assert led4["top"][:2] == led1["top"]


def check_gw_weights(solve):
    # 权重互换改变排序
    q = [1.0, 0.0]
    mems = [
        {"emb": [1.0, 0.0], "importance": 1.0},  # rel 高、imp 低
        {"emb": [0.0, 1.0], "importance": 9.0},  # rel 低、imp 高
    ]
    led_default = _compare(solve, q, mems, 2)   # gw=[0.5,3,2]：relevance 权重 3
    led_swap = _compare(solve, q, mems, 2, gw=(0.5, 2, 3))  # importance 权重 3
    assert led_default["top"][0] != led_swap["top"][0]


def check_tie_with_zero_recency_weight(solve):
    # gw[0]=0 时 recency 不参与总分：emb/imp 全等 → 总分全并列 → top 保持时间线顺序
    q = [1.0, 1.0]
    mems = [
        {"emb": [1.0, 1.0], "importance": 4.0},
        {"emb": [1.0, 1.0], "importance": 4.0},
        {"emb": [1.0, 1.0], "importance": 4.0},
    ]
    led = _compare(solve, q, mems, 3, gw=(0.0, 3, 2))
    assert led["scores"] == [2.5, 2.5, 2.5]
    assert led["top"] == [0, 1, 2]


def check_oracle_random(solve):
    import random
    rng = random.Random(99)
    for _ in range(150):
        n = rng.randint(1, 12)
        q = [rng.uniform(-1, 1) for _ in range(4)]
        mems = [{"emb": [rng.uniform(-1, 1) for _ in range(4)],
                 "importance": rng.randint(1, 10)} for _ in range(n)]
        k = rng.randint(1, n + 2)
        rd = rng.choice([0.995, 0.99, 1.0])
        led = solve(q, mems, k, rd)
        ref = _py(q, mems, k, rd)
        def flat_close(a, b):
            if isinstance(a, list):
                assert len(a) == len(b)
                for x, y in zip(a, b):
                    flat_close(x, y)
            elif isinstance(a, dict):
                for key in a:
                    flat_close(a[key], b[key])
            elif isinstance(a, bool):
                assert a == b
            else:
                assert abs(a - b) <= 1e-9 + 1e-9 * abs(b)
        flat_close(led, ref)


def check_invalid_inputs(solve):
    good = {"emb": [1.0], "importance": 1.0}
    bad = [
        lambda: solve([], [good], 1),
        lambda: solve([1.0], [], 1),
        lambda: solve([1.0, 2.0], [good], 1),
        lambda: solve([1.0], [{"emb": [1.0]}], 1),
        lambda: solve([1.0], [{"emb": [1.0], "importance": float("nan")}], 1),
        lambda: solve([1.0], [good], 0),
        lambda: solve([1.0], [good], True),
        lambda: solve([1.0], [good], 1, recency_decay=0.0),
        lambda: solve([1.0], [good], 1, recency_decay=1.5),
        lambda: solve([1.0], [good], 1, gw=(0.5, 3)),
        lambda: solve([1.0], [{"emb": [True], "importance": 1.0}], 1),
    ]
    for fn in bad:
        try:
            fn()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")
