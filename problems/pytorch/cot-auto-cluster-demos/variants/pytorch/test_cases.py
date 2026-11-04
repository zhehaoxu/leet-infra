"""Auto-CoT 聚类选代表拼 demo -- 测试用例."""


def _oracle(questions, rationales, pred_anses, assignment, dist,
            max_ra_len=5, arithmetic=True, gold_anses=None):
    """独立复刻官方 run_demo.py 的选择核心（不 import solution）。"""
    k = len(dist[0])
    clusters = {i: [] for i in range(k)}
    for sid in range(len(questions)):
        clusters[assignment[sid]].append(sid)  # sid 升序 append
    demos, selected, order = [], [], []
    for ci in range(k):
        cands = sorted(clusters[ci], key=lambda sid: dist[sid][ci])
        for sid in cands:
            q = questions[sid]
            r = rationales[sid].strip()
            p = pred_anses[sid].strip()
            if len(q.strip().split()) > 60:
                continue
            if len(r.replace("\n\n", "\n").split("\n")) > max_ra_len:
                continue
            if not r.endswith("."):
                continue
            if p == "":
                continue
            if arithmetic and not (p in r.split(".")[-2]
                                   or p in r.split()[-10:]):
                continue
            demos.append({
                "question": q,
                "rationale": " ".join(r.replace("\n\n", "\n")
                                      .replace("\n", " ").strip().split()),
                "pred_ans": p,
                "gold_ans": None if gold_anses is None
                else gold_anses[sid].strip(),
            })
            selected.append(sid)
            order.append(ci)
            break
    return {"demos": demos, "selected": selected,
            "cluster_order": order, "num_clusters": k}


def _base_fixture():
    """8 样本 + 1 个 55 词问题样本，k=5（簇 4 为空簇）。"""
    questions = [
        "Alice has three apples",                          # s0 簇0 正常
        " ".join(["word"] * 61),                           # s1 簇0 61 词
        "Bob counts two boxes",                            # s2 簇0 五行 rationale
        "Cara has four cats",                              # s3 簇0 无结尾句点
        "Dan owns five dogs",                              # s4 簇0 空答案
        "Eve sees nine birds",                             # s5 簇0 数字 9 不在文本
        "Fay keeps six fish",                              # s6 簇1 次近候选
        "Gus plants eight roses",                          # s7 簇2 正常
        "Hal reads fifty five words here",                 # s8 簇3 55 词正常
        "Ivy walks ten steps",                             # s9 簇1 最近候选
    ]
    rationales = [
        "Intro line for context. Twelve items remain and we walk through "
        "every single remaining step of this very long reasoning slowly.",
        "One single reasoning line.",
        "l1.\nl2.\nl3.\nl4.\nl5.",
        "It does not end with a period",
        "He counts them one by one and stops.",
        "We count nine items here. Nine stays nine at the end.",
        "Two groups of four. Two times four equals eight apples in the basket.",
        "Count five ducks first. Then add four ducks to reach nine in total.",
        "He opens a book. He reads until the sun sets at last.",
        "She takes the first step. She repeats it nine more times to reach "
        "ten in total.",
    ]
    pred_anses = ["Twelve", "1", "l5", "3", "", "9", "eight", "nine",
                  "sun", "ten"]
    assignment = [0, 0, 0, 0, 0, 0, 1, 2, 3, 1]
    dist = [
        [1.0, 9.0, 9.0, 9.0, 9.0],
        [2.0, 9.0, 9.0, 9.0, 9.0],
        [0.5, 9.0, 9.0, 9.0, 9.0],
        [4.0, 9.0, 9.0, 9.0, 9.0],
        [5.0, 9.0, 9.0, 9.0, 9.0],
        [6.0, 9.0, 9.0, 9.0, 9.0],
        [9.5, 0.5, 0.1, 9.9, 9.9],
        [9.5, 9.0, 0.8, 9.9, 9.9],
        [9.5, 9.0, 9.0, 0.3, 9.9],
        [9.9, 0.4, 9.9, 9.9, 9.9],
    ]
    return questions, rationales, pred_anses, assignment, dist


def _run(solve, *args, **kw):
    return solve(*args, **kw)


def check_main_oracle(solve):
    q, r, p, a, d = _base_fixture()
    exp = _oracle(q, r, p, a, d, max_ra_len=2)
    got = _run(solve, q, r, p, a, d, max_ra_len=2)
    assert got == exp, (got, exp)
    # max_ra_len=2 时簇 0 只有 s0 通过；簇 1 取更近的 s9；簇 2/3 各一个
    assert got["selected"] == [0, 9, 7, 8], got["selected"]
    assert got["cluster_order"] == [0, 1, 2, 3]
    assert got["num_clusters"] == 5
    assert len(got["demos"]) == 4
    assert got["demos"][0]["pred_ans"] == "Twelve"
    assert got["demos"][0]["gold_ans"] is None
    assert got["demos"][1]["pred_ans"] == "ten"
    # 宽松 max_ra_len=5：更近的 s2（五行 rationale）恰通过并独占簇 0 名额
    got5 = _run(solve, q, r, p, a, d)
    exp5 = _oracle(q, r, p, a, d)
    assert got5 == exp5
    assert got5["selected"] == [2, 9, 7, 8]


def check_gold_ans_passthrough(solve):
    q, r, p, a, d = _base_fixture()
    gold = [" 5 ", "1", "2", "3", "4", "9", " eight ", " nine", "words ",
            " ten "]
    got = _run(solve, q, r, p, a, d, max_ra_len=2, gold_anses=gold)
    exp = _oracle(q, r, p, a, d, max_ra_len=2, gold_anses=gold)
    assert got == exp
    assert got["demos"][0]["gold_ans"] == "5"
    assert got["demos"][1]["gold_ans"] == "ten"
    assert got["demos"][2]["gold_ans"] == "nine"


def check_word_count_filter(solve):
    # 55 词问题：<=60 通过、>50 变异路径被题面判据抓（这里验官方语义两端）
    q55 = " ".join(["token"] * 55)
    q61 = " ".join(["token"] * 61)
    qs = [q55, q61, "short one"]
    rs = ["He reads one line. Then he stops for good.",
          "Another fine rationale line.", "A clean single line."]
    ps = ["good", "line", "clean"]
    a = [0, 1, 2]
    d = [[0.4, 9.0, 9.0], [9.0, 0.6, 9.0], [9.0, 9.0, 0.7]]
    got = _run(solve, qs, rs, ps, a, d)
    # q55（55 词）通过；q61（61 词）被上限过滤；短句通过
    assert got["selected"] == [0, 2]
    # 同 fixture 的 oracle 对拍
    assert got == _oracle(qs, rs, ps, a, d)


def check_ra_len_boundary(solve):
    # 行数恰等于 max_ra_len 通过、超一行不过；\n\n 折叠后计数
    r3 = "first line here.\n\nsecond line now.\nthird line ends."
    # replace("\n\n", "\n") 后 3 行
    qs = ["one two", "one two", "one two"]
    rs = [r3, r3, "single line."]
    ps = ["first", "first", "single"]
    d = [[0.5, 9.0], [0.6, 9.0], [9.0, 0.7]]
    a = [0, 0, 1]
    got = _run(solve, qs, rs, ps, a, d, max_ra_len=3)
    assert got["selected"] == [0, 2]
    got2 = _run(solve, qs, rs, ps, a, d, max_ra_len=2)
    assert got2["selected"] == [2] and got2["cluster_order"] == [1]
    assert got2["demos"][0]["pred_ans"] == "single"


def check_arithmetic_verify(solve):
    # 四候选各占一簇：s0 命中 [-2]；s1 命中尾 10 词但不中 [-2]；
    # s2 答案（数字 9）完全不在文本；s3 大写 "Plain" 恰在单句 [-2] 中
    qs = ["a one", "b two", "c three", "d four"]
    rs = [
        "First count the items. Total equals twelve apples overall.",
        "We start with a pile. Then subtract one. Then we keep eight here. "
        "Now five remain at the end here.",
        "Nine items are here. We then remove three of them. "
        "Next we double the rest. Finally we sort and count everything "
        "carefully one more time.",
        "Plain rationale without digits at all inside.",
    ]
    ps = [" twelve ", "eight", "9", "Plain"]
    a = [0, 1, 2, 3]
    d = [[0.5, 9.0, 9.0, 9.0], [9.0, 0.6, 9.0, 9.0],
         [9.0, 9.0, 0.7, 9.0], [9.0, 9.0, 9.0, 0.9]]
    got = _run(solve, qs, rs, ps, a, d)
    assert got["selected"] == [0, 1, 3], got
    # arithmetic=False：s2（答案不在文本）也放行
    got_off = _run(solve, qs, rs, ps, a, d, arithmetic=False)
    assert got_off["selected"] == [0, 1, 2, 3], got_off
    assert got_off == _oracle(qs, rs, ps, a, d, arithmetic=False)


def check_tie_and_empty(solve):
    # 距离平局取小下标；簇 2 为空簇无 demo
    qs = ["x one", "y two", "z three"]
    rs = ["First line here. Then it ends well.", "Another line or two. Fine.",
          "Last one is fine too."]
    ps = ["well", "two", "fine"]
    a = [0, 0, 1]
    d = [[0.5, 9.0], [0.5, 9.0], [9.0, 0.2]]
    # 簇 2 编号存在但无样本 -> num_clusters=3、无簇 2 demo
    d0 = [[0.5, 9.0, 9.0], [0.5, 9.0, 9.0], [9.0, 9.0, 0.2]]
    got = _run(solve, qs, rs, ps, [0, 0, 1], d0)
    assert got["selected"] == [0, 2]
    assert got["cluster_order"] == [0, 1]
    assert got["num_clusters"] == 3
    assert got == _oracle(qs, rs, ps, [0, 0, 1], d0)


def check_folding_cleanup(solve):
    qs = ["one two three"]
    rs = ["Step  one here.\n\nStep   two now.\nend line."]
    ps = ["Step"]
    d = [[0.5]]
    got = _run(solve, qs, rs, ps, [0], d)
    # 官方清洗：\n\n→\n→空格、strip、空白折叠单空格
    assert got["demos"][0]["rationale"] == "Step one here. Step two now. end line."
    assert got["demos"][0]["question"] == "one two three"
    assert got["demos"][0]["pred_ans"] == "Step"
    # 行数按 \n 计数：折叠前 3 行 <= max_ra_len=3
    got1 = _run(solve, qs, rs, ps, [0], d, max_ra_len=1)
    assert got1["selected"] == []


def check_purity(solve):
    import copy
    q, r, p, a, d = _base_fixture()
    gold = ["5"] * 10
    snap = (copy.deepcopy(q), copy.deepcopy(r), copy.deepcopy(p),
            copy.deepcopy(a), copy.deepcopy(d), copy.deepcopy(gold))
    _run(solve, q, r, p, a, d, max_ra_len=2, gold_anses=gold)
    assert (q, r, p, a, d, gold) == snap


def check_args_validation(solve):
    q, r, p, a, d = _base_fixture()
    ok = lambda *args, **kw: _run(solve, *args, **kw)
    for bad in (
        lambda: ok("nope", r, p, a, d),
        lambda: ok([], [], [], [], []),
        lambda: ok([1], ["x."], ["x"], [0], [[0.5]]),
        lambda: ok(q[:8], r, p, a, d),
        lambda: ok(q, r[:8], p, a, d),
        lambda: ok(q, r, p[:8], a, d),
        lambda: ok(q, r, p, a[:8], d),
        lambda: ok(q, r, p, [True] * 10, d),
        lambda: ok(q, r, p, ["0"] * 10, d),
        lambda: ok(q, r, p, [5] * 10, d),
        lambda: ok(q, r, p, a, "nope"),
        lambda: ok(q, r, p, a, d[:8]),
        lambda: ok(q, r, p, a, [[0.5]] * 9),
        lambda: ok(q, r, p, a, [[]] * 9),
        lambda: ok(q, r, p, a, [[0.5, 0.5]] + d[1:]),
        lambda: ok(q, r, p, a, [["x"]] * 9),
        lambda: ok(q, r, p, [0] * 10, [["x", 0.5, 0.5, 0.5, 0.5]]
                   + [[0.5] * 5] * 9),
        lambda: ok(q, r, p, [0] * 10, [(0.5,) * 5] * 10),
        lambda: ok(q, r, p, [0] * 10, [[0.5] * 3] + [[0.5] * 5] * 9),
        lambda: ok(q, r, p, a, d, max_ra_len=0),
        lambda: ok(q, r, p, a, d, max_ra_len=True),
        lambda: ok(q, r, p, a, d, arithmetic="yes"),
        lambda: ok(q, r, p, a, d, gold_anses=["x"] * 8),
    ):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")
    try:
        ok(q, r, p, a, d, max_ra_len=0)
    except ValueError as e:
        assert "max_ra_len" in str(e), e
    try:
        ok(q, r, p, [5] * 10, d)
    except ValueError as e:
        assert "越界" in str(e), e


check_main_oracle.example = True
check_gold_ans_passthrough.example = True
check_arithmetic_verify.example = True
check_ra_len_boundary.example = True
check_word_count_filter.example = True
check_tie_and_empty.example = True
check_folding_cleanup.example = True
check_purity.example = True
check_args_validation.example = True
