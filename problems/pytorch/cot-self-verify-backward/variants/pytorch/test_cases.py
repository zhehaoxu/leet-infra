"""Self-Verification 后向验证 -- 测试用例."""

import math


def _normalize(s):
    try:
        return str(float(s))
    except Exception:
        return s


def _oracle(question, sampled_answers, verifier_outputs, ground_truth,
            declarative=""):
    """独立复刻官方账本（rfind 路径 + Counter 思路）。"""
    answers = []
    for s in sampled_answers:
        item = _normalize(s)
        if item != "" and item not in answers:
            answers.append(item)
    gt = _normalize(ground_truth)

    def make_declarative(q, a):
        return "{} The answer is {}.".format(q, a)

    scores = {}
    sentences = []
    sub_numbers = []
    if len(answers) == 1:
        scores = {0: 1}
    else:
        for a_idx, ans_a in enumerate(answers):
            text = question.replace(",", ".")
            pos = text[::-1].find(".")
            q_tail = text[len(text) - pos:]
            ts = text[:len(text) - pos]
            if declarative == "":
                decl = make_declarative(q_tail, ans_a)
            elif answers[0] in declarative:
                i = declarative.rfind(answers[0])
                decl = declarative[:i] + ans_a + declarative[i + len(answers[0]):]
            else:
                decl = make_declarative(q_tail, ans_a)
            numbers = []
            i = 0
            while i < len(ts):
                m = _match_number(ts, i)
                if m:
                    numbers.append(m)
                    i += len(m)
                else:
                    i += 1
            sent_list = []
            for nqn in numbers:
                new_ts = ""
                for seg in ts.split("."):
                    if nqn in seg:
                        p = seg.find(nqn)
                        new_ts += seg[:p] + "'X'" + seg[p + len(nqn):] + "."
                    else:
                        new_ts += seg + "."
                new_ts = new_ts.replace("..", ".")
                if new_ts[0] == " ":
                    new_ts = new_ts[1:]
                if new_ts[-1] != " ":
                    new_ts += " "
                new_ts = new_ts.replace(" .", ".")
                sent_list.append('"' + new_ts + decl + '"')
            sent_list = sent_list[:3]
            numbers = numbers[:3]
            total = 0
            for d, nqn in enumerate(numbers):
                verdicts = verifier_outputs[a_idx * 3 + d] \
                    if a_idx * 3 + d < len(verifier_outputs) else []
                conv = []
                for v in verdicts:
                    try:
                        conv.append(float(v))
                    except Exception:
                        conv.append(v)
                try:
                    expected = float(nqn)
                except Exception:
                    expected = nqn
                total += sum(1 for v in conv if v == expected)
            scores[a_idx] = total
            sentences.append(sent_list)
            sub_numbers.append(numbers)

    chosen_idx = 0
    best = None
    for k in sorted(scores):
        if best is None or scores[k] > best:
            best = scores[k]
            chosen_idx = k
    return {
        "answers": answers,
        "sentences": sentences,
        "sub_numbers": sub_numbers,
        "scores": scores,
        "chosen_idx": chosen_idx,
        "chosen_answer": answers[chosen_idx],
        "is_true": [a == gt for a in answers],
        "chosen_correct": answers[chosen_idx] == gt,
    }


def _match_number(text, i):
    """独立实现的 -?\\d+\\.?\\d* 扫描（替代 regex 路径）。"""
    j = i
    if j < len(text) and text[j] == "-":
        j += 1
    start_digits = j
    while j < len(text) and text[j].isdigit():
        j += 1
    if j == start_digits:
        return None
    if j < len(text) and text[j] == ".":
        j += 1
        while j < len(text) and text[j].isdigit():
            j += 1
    return text[i:j]


QUESTION = "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. Jason gave Denny 6 lollipops."


def _approx(a, b):
    if isinstance(a, float) and math.isnan(a):
        return isinstance(b, float) and math.isnan(b)
    return a == b


def check_candidate_collection(solve):
    got = solve(QUESTION, ["20", "5.0", "20", "5", "", "six"],
                [], "5", declarative="")
    # float 规范化 "20"→"20.0"、"5.0"/"5"→"5.0" 去重、空串剔除、非数值原样
    assert got["answers"] == ["20.0", "5.0", "six"], got["answers"]
    assert got["is_true"] == [False, True, False]
    # 单候选路径
    got1 = solve(QUESTION, ["42"], [], "42.0")
    assert got1["answers"] == ["42.0"]
    assert got1["scores"] == {0: 1}
    assert got1["sentences"] == []
    assert got1["chosen_correct"] is True
    # 候选唯一但错
    got2 = solve(QUESTION, ["42"], [], "41")
    assert got2["chosen_correct"] is False


def check_subquestion_blank(solve):
    q = "Jason had 20 lollipops. He gave Denny 6 lollipops."
    # 数字 20/6；候选 5.0 挖 20/6，候选 6.0 挖 20/6；
    # verifier 是**数字版**（demo_B = create_verifier_demo_text）：
    # 判定串输出挖空处的数值，与原数字一致才计分。
    # 扁平池索引 a_idx*3+d：候选 0 用 [0][1]、候选 1 用 [3][4]（[2] 是死槽）
    got = solve(q, ["5", "6"], [["20"] * 10, ["7"] * 10, ["0"] * 3,
                                ["20"] * 10, ["6"] * 10],
                "6", declarative="Jason had X lollipops. The answer is 5.")
    assert got["answers"] == ["5.0", "6.0"]
    assert got["sub_numbers"][0] == ["20", "6"]
    s0 = got["sentences"][0]
    # 挖 20：前句变 'X'，次句不动；挖 6：前句不动，次句变 'X'
    assert "'X' lollipops. He gave Denny 6" in s0[0]
    assert "Jason had 20 lollipops. He gave Denny 'X'" in s0[1]
    assert got["sub_numbers"][1] == ["20", "6"]
    # 候选 0：子问题 0 期望 20（["20"]*10 → 10 分）、子问题 1 期望 6（["7"] → 0）
    # 候选 1：20 一致 10 分 + 6 一致 10 分
    assert got["scores"] == {0: 10, 1: 20}
    assert got["chosen_idx"] == 1 and got["chosen_correct"] is True
    exp = _oracle(q, ["5", "6"], [["20"] * 10, ["7"] * 10, ["0"] * 3,
                                  ["20"] * 10, ["6"] * 10],
                  "6", declarative="Jason had X lollipops. The answer is 5.")
    assert got["sentences"] == exp["sentences"]
    assert got["scores"] == exp["scores"]


def check_top3_and_numbers(solve):
    # 官方正则 \\d+\\.?\\d* 对 "subtract 1. Done" 抽出 "1."（点后无数字也吃）；
    # 跨句点的数字（-2.5 被句点切开）找不到含它的句段 → 挖空不发生（官方陷阱）
    q = "Start at -2.5 here. Add 3 apples. Then 2 boxes. Finally 1 car. Done."
    got = solve(q, ["8", "7"], [], "7", declarative="")
    assert got["sub_numbers"][0] == ["-2.5", "3", "2"], got["sub_numbers"]
    assert len(got["sentences"][0]) == 3
    # 挖 -2.5：跨句点 → 无 'X'（句段保持原样）；挖 3/2：正常替换
    assert "'X'" not in got["sentences"][0][0]
    assert "'X' apples" in got["sentences"][0][1]
    assert "'X' boxes" in got["sentences"][0][2]
    for s in got["sentences"][0]:
        assert ".." not in s
        assert s.startswith('"') and s.endswith('"')
    exp = _oracle(q, ["8", "7"], [], "7")
    assert got["sub_numbers"] == exp["sub_numbers"]
    assert got["sentences"] == exp["sentences"]


def check_multi_segment_replace(solve):
    # 同一数字出现在多个句段：官方 number_find 缺失 → 每个句段都替换；
    # 且数字逐处出现逐条目（重复数字产生两个相同子问题）
    q = "Tom has 3 apples. Tom has 3 oranges. Total is 6."
    got = solve(q, ["9", "6"], [], "6", declarative="")
    assert got["sub_numbers"][0] == ["3", "3", "6."], got["sub_numbers"]
    s0 = got["sentences"][0][0]  # 挖 3：两个句段都替换
    assert s0.count("'X'") == 2, s0
    assert got["sentences"][0][1] == s0  # 重复数字 → 相同子问题
    s2 = got["sentences"][0][2]  # 挖 "6."：跨句点（6. 的点被切）→ 不替换
    assert "'X'" not in s2 and "Total is 6" in s2
    exp = _oracle(q, ["9", "6"], [], "6")
    assert got["sentences"] == exp["sentences"]


def check_declarative_paths(solve):
    q = "A box has 5 balls. 2 more are added. How many now?"
    # declarative 含 answers[0]（"5.0"）：最后一次出现被替换为当前候选
    decl = "A box has 5.0 balls. The answer is 5.0."
    got = solve(q, ["5", "8"], [], "8", declarative=decl)
    assert got["sentences"][1][0].endswith("The answer is 8.0.\""), \
        got["sentences"][1][0]
    assert "5.0 balls" in got["sentences"][1][0]
    # declarative 不含 answers[0]：fallback 模板
    got2 = solve(q, ["5", "8"], [], "8",
                 declarative="Unrelated text here.")
    assert "The answer is 5.0." in got2["sentences"][0][0]
    assert got2["sentences"][0][0].startswith('"A box has \'X\' balls')
    # declarative 为空：fallback 模板（question 尾段）
    got3 = solve(q, ["5", "8"], [], "8")
    assert "The answer is 5.0." in got3["sentences"][0][0]
    exp = _oracle(q, ["5", "8"], [], "8")
    assert got3["sentences"] == exp["sentences"]


def check_scoring_and_argmax(solve):
    q = "Num is 4 here. Num is 9 there."
    # 扁平池 a_idx*3+d：候选 0 → [0][1]，候选 1 → [3][4]（[2] 死槽）
    # 候选 0（4.0）：挖 4 期望 4（["4"]*6 → 6 分）、挖 9 期望 9（["9"]*3 → 3 分）
    # 候选 1（9.0）：挖 4（["4"]*2 → 2 分）、挖 9（["9"]*9 → 9 分）
    v = [["4"] * 6, ["9"] * 3, ["0"] * 3, ["4"] * 2, ["9"] * 9]
    got = solve(q, ["4", "9"], v, "9")
    assert got["scores"] == {0: 9, 1: 11}, got["scores"]
    assert got["chosen_idx"] == 1
    # 平局取最前（np.argmax 语义）
    v_tie = [["4"] * 6, ["9"] * 3, ["0"] * 3, ["9"] * 2, ["9"] * 9]
    got_tie = solve(q, ["4", "9"], v_tie, "9")
    assert got_tie["scores"] == {0: 9, 1: 9}
    assert got_tie["chosen_idx"] == 0
    # 反向平局
    v_tie2 = [["4"] * 6, ["9"] * 3, ["0"] * 3, ["4"] * 2, ["9"] * 7]
    got_tie2 = solve(q, ["4", "9"], v_tie2, "9")
    assert got_tie2["scores"] == {0: 9, 1: 9}
    assert got_tie2["chosen_idx"] == 0
    # 混合判定串：可 float 的按 float 比，不可的按字符串比
    v4 = [["4.0"] * 3, ["9", "x", "4.0"], ["0"] * 3, ["9.0"] * 4, ["y", "9.0"]]
    got4 = solve(q, ["4", "9"], v4, "9")
    exp4 = _oracle(q, ["4", "9"], v4, "9")
    assert got4["scores"] == exp4["scores"]
    # 缺失判定池按 0 分
    got5 = solve(q, ["4", "9"], [["4"] * 6], "9")
    assert got5["scores"] == {0: 6, 1: 0}


def check_comma_and_tail(solve):
    # 官方把逗号先转句点：3,000 → "3.000" 抽出 "3.000"
    q = "Total was 3,000 last year. It is 4,000 now. Diff is 1,000."
    got = solve(q, ["1", "2"], [], "2", declarative="")
    assert got["sub_numbers"][0][:2] == ["3.000", "4.000"], got["sub_numbers"]
    exp = _oracle(q, ["1", "2"], [], "2")
    assert got["sub_numbers"] == exp["sub_numbers"]
    assert got["sentences"] == exp["sentences"]
    # 尾空段产生的双句点被 '..' → '.' 修正
    for s in got["sentences"][0]:
        assert ".." not in s


def check_purity(solve):
    import copy
    q = QUESTION
    sa = ["20", "5.0", "20"]
    vo = [["True"] * 3] * 6
    decl = "Jason had 20.0 lollipops."
    snap = (copy.deepcopy(q), copy.deepcopy(sa), copy.deepcopy(vo),
            copy.deepcopy(decl))
    solve(q, sa, vo, "5", declarative=decl)
    assert (q, sa, vo, decl) == snap


def check_args_validation(solve):
    for bad in (
        lambda: solve("", ["1"], [], "1"),
        lambda: solve(5, ["1"], [], "1"),
        lambda: solve(QUESTION, [], [], "1"),
        lambda: solve(QUESTION, "nope", [], "1"),
        lambda: solve(QUESTION, [1], [], "1"),
        lambda: solve(QUESTION, ["1"], "nope", "1"),
        lambda: solve(QUESTION, ["1", "2"], ["x"], "1"),
        lambda: solve(QUESTION, ["1", "2"], [1], "1"),
        lambda: solve(QUESTION, ["1", "2"], [["x"], [2]], "1"),
        lambda: solve(QUESTION, ["1"], [], 1),
        lambda: solve(QUESTION, ["1"], [], "1", declarative=5),
    ):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")
    try:
        solve(QUESTION, [], [], "1")
    except ValueError as e:
        assert "sampled_answers" in str(e), e
    try:
        solve(QUESTION, ["1"], [1], "1")
    except ValueError as e:
        assert "verifier_outputs" in str(e), e


check_candidate_collection.example = True
check_subquestion_blank.example = True
check_top3_and_numbers.example = True
check_multi_segment_replace.example = True
check_declarative_paths.example = True
check_scoring_and_argmax.example = True
check_comma_and_tail.example = True
check_purity.example = True
check_args_validation.example = True
