def _oracle(matches, models):
    """独立重写的 oracle：解析用正则搜索列表改写、结算用枚举组合表、
    聚合用 defaultdict 累加——与 solution 不同代码路径，同官方语义。"""
    import re
    from collections import defaultdict

    # 官方 if/elif 链 = 标记优先级（[[A]] 恒先于 [[B]]/[[C]] 检查，与位置无关）
    marks = ["[[A]]", "[[B]]", "[[C]]"]

    def parse(text):
        for i, k in enumerate(marks):
            if k in text:
                return {0: "A", 1: "B", 2: "tie"}[i]
        return "error"

    rows = []
    tally = defaultdict(lambda: [0, 0, 0])  # win, loss, tie
    for m in matches:
        w1 = parse(m["g1_judgment"])
        w2 = parse(m["g2_judgment"])
        # swap 账：g2 的 A 位摆的是 answer_2
        g1 = m["model_1"] if w1 == "A" else m["model_2"] if w1 == "B" else w1
        g2 = m["model_2"] if w2 == "A" else m["model_1"] if w2 == "B" else w2
        if "error" in (g1, g2):
            outcome = "error"
        elif g1 == "tie" or g1 != g2:
            outcome = "tie"
        else:
            outcome = g1
        rows.append({
            "question_id": m["question_id"],
            "g1_winner": g1,
            "g2_winner": g2,
            "outcome": outcome,
        })
        if outcome == "error":
            continue
        if outcome == "tie":
            tally[m["model_1"]][2] += 1
            tally[m["model_2"]][2] += 1
        elif outcome == m["model_1"]:
            tally[m["model_1"]][0] += 1
            tally[m["model_2"]][1] += 1
        else:
            tally[m["model_2"]][0] += 1
            tally[m["model_1"]][1] += 1

    stats = {}
    for name in models:
        w, l, t = tally.get(name, [0, 0, 0])
        total = w + l + t
        stats[name] = {
            "win": w,
            "loss": l,
            "tie": t,
            "win_rate_adjusted": (w + 0.5 * t) / total if total else 0.0,
        }
    return {"matches": rows, "stats": stats}


def _approx(x, y, tol=1e-9):
    return abs(x - y) <= tol


def _compare(solve, matches, models):
    exp = _oracle(matches, models)
    got = solve(matches, models)
    assert set(got) == {"matches", "stats"}, f"键不一致：{sorted(got)}"
    assert len(got["matches"]) == len(exp["matches"])
    for i, (a, b) in enumerate(zip(got["matches"], exp["matches"])):
        assert a == b, f"matches[{i}]：期望 {b}，得到 {a}"
    assert set(got["stats"]) == set(exp["stats"]), (
        f"stats 模型集：期望 {sorted(exp['stats'])}，得到 {sorted(got['stats'])}"
    )
    for name, e in exp["stats"].items():
        g = got["stats"][name]
        for k in ("win", "loss", "tie"):
            assert g[k] == e[k], f"stats[{name}][{k}]：期望 {e[k]}，得到 {g[k]}"
        assert _approx(g["win_rate_adjusted"], e["win_rate_adjusted"]), (
            f"stats[{name}].win_rate_adjusted：期望 {e['win_rate_adjusted']}，得到 {g['win_rate_adjusted']}"
        )
    return got


def check_consistent_win(solve):
    # 两局一致：g1 [[A]]（A 位 = model_1）→ model_1；
    # g2 摆位交换后 [[B]]（B 位 = answer_1 = model_1）→ 仍是 model_1 → 真胜
    matches = [{
        "question_id": 1, "model_1": "alpha", "model_2": "beta",
        "g1_judgment": "The better answer is [[A]].",
        "g2_judgment": "After swapping, I choose [[B]].",
    }]
    led = _compare(solve, matches, ["alpha", "beta"])
    r = led["matches"][0]
    assert r["g1_winner"] == "alpha" and r["g2_winner"] == "alpha"
    assert r["outcome"] == "alpha"
    assert led["stats"]["alpha"]["win"] == 1
    assert led["stats"]["beta"]["loss"] == 1
    assert _approx(led["stats"]["alpha"]["win_rate_adjusted"], 1.0)
    assert _approx(led["stats"]["beta"]["win_rate_adjusted"], 0.0)


def check_position_bias_disagreement(solve):
    # 位置偏差的典型形态：swap 后裁判仍判 A 位 → 两局各判一方 → 平局
    # g1: A=model_1 胜；g2: A=model_2 胜 → 不一致 → 双 tie
    matches = [{
        "question_id": 7, "model_1": "alpha", "model_2": "beta",
        "g1_judgment": "x [[A]] y",
        "g2_judgment": "y [[A]] x",
    }]
    led = _compare(solve, matches, ["alpha", "beta"])
    r = led["matches"][0]
    assert r["g1_winner"] == "alpha" and r["g2_winner"] == "beta"
    assert r["outcome"] == "tie"
    assert led["stats"]["alpha"]["tie"] == 1 and led["stats"]["beta"]["tie"] == 1
    # 平局算半胜：0.5/1
    assert _approx(led["stats"]["alpha"]["win_rate_adjusted"], 0.5)


def check_any_tie_is_tie(solve):
    # 官方条件是 "g1 == tie or g1 != g2"：g1 平局则整场平局，
    # 即便 g2 分出了胜负（被丢弃）
    matches = [{
        "question_id": 2, "model_1": "alpha", "model_2": "beta",
        "g1_judgment": "both good [[C]]",
        "g2_judgment": "clearly [[A]]",
    }]
    led = _compare(solve, matches, ["alpha", "beta"])
    r = led["matches"][0]
    assert r["g1_winner"] == "tie" and r["g2_winner"] == "beta"
    assert r["outcome"] == "tie"


def check_error_dropped(solve):
    # 任一局解析失败 → 整场剔除，不计入任何统计
    matches = [
        {"question_id": 3, "model_1": "alpha", "model_2": "beta",
         "g1_judgment": "I like the first one.",   # 无标记
         "g2_judgment": "winner [[B]]"},
        {"question_id": 4, "model_1": "alpha", "model_2": "beta",
         "g1_judgment": "[[A]]", "g2_judgment": "[[B]]"},
    ]
    led = _compare(solve, matches, ["alpha", "beta"])
    assert led["matches"][0]["outcome"] == "error"
    assert led["matches"][1]["outcome"] == "alpha"
    # 只有第二场计入
    assert led["stats"]["alpha"]["win"] == 1
    assert led["stats"]["beta"]["loss"] == 1


def check_both_error_and_absent_models(solve):
    # 全 error 场次 → 无统计行；缺席模型给全 0 行、win_rate_adjusted = 0.0
    matches = [{"question_id": 5, "model_1": "alpha", "model_2": "beta",
                "g1_judgment": "no marks", "g2_judgment": "none either"}]
    led = _compare(solve, matches, ["alpha", "beta", "gamma"])
    assert led["matches"][0]["outcome"] == "error"
    for name in ("alpha", "beta", "gamma"):
        s = led["stats"][name]
        assert (s["win"], s["loss"], s["tie"]) == (0, 0, 0)
        assert _approx(s["win_rate_adjusted"], 0.0)


def check_mark_priority_quirk(solve):
    # 官方 if/elif 顺序 quirk：同一文本里 [[A]] 先于 [[B]] 检查，
    # 先命中先判（哪怕语境上另一个标记更"合理"）
    matches = [
        {"question_id": 6, "model_1": "alpha", "model_2": "beta",
         "g1_judgment": "first [[A]] then [[B]]",   # A 优先 → model_1
         "g2_judgment": "second [[B]] and [[A]]"},  # A 优先 → model_2
    ]
    led = _compare(solve, matches, ["alpha", "beta"])
    r = led["matches"][0]
    assert r["g1_winner"] == "alpha" and r["g2_winner"] == "beta"
    assert r["outcome"] == "tie"


def check_case_sensitive_and_c_tie(solve):
    # 大小写敏感："[[a]]" 不算标记；[[C]] 是平局
    matches = [
        {"question_id": 8, "model_1": "alpha", "model_2": "beta",
         "g1_judgment": "lower [[a]] no", "g2_judgment": "[[C]]"},
        {"question_id": 9, "model_1": "alpha", "model_2": "beta",
         "g1_judgment": "[[C]]", "g2_judgment": "[[C]]"},
    ]
    led = _compare(solve, matches, ["alpha", "beta"])
    assert led["matches"][0]["outcome"] == "error"   # g1 解析失败
    assert led["matches"][1]["outcome"] == "tie"


def check_error_beats_tie(solve):
    # 隐蔽优先级：error 剔除先于 tie 条件 —— g1 平局但 g2 解析失败时，
    # 整场仍被剔除（官方 show_result 先过滤 error 行，tie 判定轮不到）
    matches = [{"question_id": 14, "model_1": "alpha", "model_2": "beta",
                "g1_judgment": "[[C]]", "g2_judgment": "no marks here"}]
    led = _compare(solve, matches, ["alpha", "beta"])
    assert led["matches"][0]["g1_winner"] == "tie"
    assert led["matches"][0]["g2_winner"] == "error"
    assert led["matches"][0]["outcome"] == "error"
    assert led["stats"]["alpha"]["tie"] == 0


def check_aggregation_over_matches(solve):
    # 多场次聚合：win/loss/tie 累计，win_rate_adjusted = (win + 0.5*tie) / total
    # alpha: 2 胜 1 平 0 负（对 beta），beta: 0 胜 2 负 1 平
    matches = [
        {"question_id": 10, "model_1": "alpha", "model_2": "beta",
         "g1_judgment": "[[A]]", "g2_judgment": "[[B]]"},        # alpha 胜
        {"question_id": 11, "model_1": "beta", "model_2": "alpha",
         "g1_judgment": "[[B]]", "g2_judgment": "[[A]]"},        # 双指 alpha
        {"question_id": 12, "model_1": "alpha", "model_2": "beta",
         "g1_judgment": "[[C]]", "g2_judgment": "both fine [[C]]"},  # 双平
    ]
    led = _compare(solve, matches, ["alpha", "beta"])
    sa, sb = led["stats"]["alpha"], led["stats"]["beta"]
    assert (sa["win"], sa["loss"], sa["tie"]) == (2, 0, 1)
    assert (sb["win"], sb["loss"], sb["tie"]) == (0, 2, 1)
    assert _approx(sa["win_rate_adjusted"], (2 + 0.5) / 3)
    assert _approx(sb["win_rate_adjusted"], (0 + 0.5) / 3)
    # 第二场 model_1=beta 输：g1 [[B]]→model_2=alpha、g2 [[A]]→model_2=alpha
    assert led["matches"][1]["outcome"] == "alpha"


def check_loss_from_swapped_positions(solve):
    # 负方向 swap 账：g1 A=model_1 但两局一致指 model_2 → model_1 记 loss
    matches = [{
        "question_id": 13, "model_1": "alpha", "model_2": "beta",
        "g1_judgment": "I pick [[B]] here",
        "g2_judgment": "still [[A]] after swap",
    }]
    led = _compare(solve, matches, ["alpha", "beta"])
    r = led["matches"][0]
    # g1 B→model_2；g2 A→model_2 → 一致 → beta 胜
    assert r["g1_winner"] == "beta" and r["g2_winner"] == "beta"
    assert r["outcome"] == "beta"
    assert led["stats"]["alpha"]["loss"] == 1
    assert led["stats"]["beta"]["win"] == 1
