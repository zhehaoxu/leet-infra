"""TTRL 多数票伪奖励账的检查点。

每个检查点先拿一份**另一套写法**的 oracle（栈式括号配平、先收集次序再按
`(票数, -首次位置)` 排序取冠军、指标从零重算）与 `solve` 的全量账本比对，
再补硬断言钉住容易写错的口径。
"""


# ---------------------------------------------------------------- oracle --
def _o_extract(text):
    if "\\boxed" not in text:
        return None
    start = text.rfind("\\boxed")
    stack = []
    for pos in range(start, len(text)):
        char = text[pos]
        if char == "{":
            stack.append(pos)
        elif char == "}":
            if not stack:
                return None
            stack.pop()
            if not stack:
                piece = text[start:pos + 1]
                if piece[:7] != "\\boxed{" or piece[-1] != "}":
                    return None
                return piece[7:-1]
    return None


def _o_canon(value, table):
    if table is None:
        return value
    if value in table:
        return table[value]
    return value


def _o_vote(answers, size):
    order, counts = [], {}
    for answer in answers:
        if answer not in counts:
            counts[answer] = 0
            order.append(answer)
        counts[answer] += 1
    if not order:
        return "None", 0.0, []
    seat = {answer: pos for pos, answer in enumerate(order)}
    winner = max(order, key=lambda a: (counts[a], -seat[a]))
    return winner, counts[winner] / size, [[a, counts[a]] for a in order]


def _o_oracle(outputs, n, original_gt, canon=None, keep=None):
    outputs = list(outputs)
    original_gt = list(original_gt)
    if n < 1:
        raise ValueError("n")
    if len(outputs) % n:
        raise ValueError("divisible")

    picked = None
    if keep is not None:
        if keep < 1:
            raise ValueError("keep")
        picked = []
        for block in range(len(outputs) // n):
            for step in range(keep):
                where = block * n + step
                if where >= len(outputs):
                    raise IndexError("list index out of range")
                picked.append(where)
        outputs = [outputs[where] for where in picked]
        n = keep

    if len(outputs) == 0:
        raise ValueError("empty")
    prompts = len(outputs) // n
    if len(original_gt) != prompts:
        raise ValueError("gt length")

    report, ratios = [], []
    for block in range(prompts):
        chunk = outputs[block * n:(block + 1) * n]
        raw = [_o_extract(text) for text in chunk]
        voted = [_o_canon(item, canon) for item in raw if item is not None]
        label, ratio, tally = _o_vote(voted, n)
        ratios.append(ratio)

        hit_majority, hit_truth = [], []
        truth = None
        if "\\boxed" in original_gt[block]:
            truth = _o_extract(original_gt[block])
        else:
            truth = original_gt[block]
        if truth is not None:
            truth = _o_canon(truth, canon)

        for item in raw:
            if item is None:
                hit_majority.append(0.0)
                hit_truth.append(0.0)
                continue
            norm = _o_canon(item, canon)
            hit_majority.append(1.0 if norm == label else 0.0)
            hit_truth.append(1.0 if norm == truth else 0.0)

        matched = sum(1 for a, b in zip(hit_majority, hit_truth) if a == b)
        report.append({
            "extracted": raw,
            "answers": voted,
            "tally": tally,
            "label": label,
            "ratio": ratio,
            "rewards": hit_majority,
            "truth_rewards": hit_truth,
            "metrics": {
                "label_accuracy":
                    1.0 if (truth is not None and label == truth) else 0.0,
                "reward_accuracy": matched / n,
                "majority_voting_reward": sum(hit_majority) / n,
                "ground_truth_reward": sum(hit_truth) / n,
                "pass@%d" % (n,): 1.0 if sum(hit_truth) >= 1 else 0.0,
            },
        })

    keys = list(report[0]["metrics"])
    metrics = {}
    for key in keys:
        metrics[key] = sum(p["metrics"][key] for p in report) / len(report)
    metrics["majority_ratio"] = sum(ratios) / len(ratios)

    return {
        "n": n,
        "num_prompts": prompts,
        "prompts": report,
        "metrics": metrics,
        "selected": picked,
    }


def _cmp(got, want, where="root"):
    if isinstance(want, float) or isinstance(got, float):
        assert isinstance(got, (int, float)) and isinstance(want, (int, float)), \
            "%s: %r vs %r" % (where, got, want)
        assert abs(got - want) <= 1e-12, "%s: %r vs %r" % (where, got, want)
        return
    if isinstance(want, dict):
        assert isinstance(got, dict), "%s: not a dict (%r)" % (where, got)
        assert sorted(got) == sorted(want), \
            "%s: keys %r vs %r" % (where, sorted(got), sorted(want))
        for key in want:
            _cmp(got[key], want[key], where + "/" + str(key))
        return
    if isinstance(want, (list, tuple)):
        assert isinstance(got, (list, tuple)), "%s: not a list (%r)" % (where, got)
        assert len(got) == len(want), \
            "%s: len %d vs %d" % (where, len(got), len(want))
        for pos, (a, b) in enumerate(zip(got, want)):
            _cmp(a, b, where + "[%d]" % (pos,))
        return
    assert got == want, "%s: %r vs %r" % (where, got, want)


def _same(solve, outputs, n, original_gt, canon=None, keep=None):
    got = solve(outputs, n, original_gt, canon=canon, keep=keep)
    want = _o_oracle(outputs, n, original_gt, canon=canon, keep=keep)
    _cmp(got, want)
    return got


def _box(answer, head="so the answer is "):
    return head + "\\boxed{" + answer + "}"


# ---------------------------------------------------------------- checks --
def check_example(solve):
    """题面示例：4 条采样、2 票对 1 票、1 条抽不出答案。"""
    r = _same(solve, [_box("5"), _box("5"), _box("6"), "no box at all"],
              4, ["5"])
    assert r["n"] == 4 and r["num_prompts"] == 1
    assert r["selected"] is None
    p = r["prompts"][0]
    assert p["extracted"] == ["5", "5", "6", None]
    assert p["answers"] == ["5", "5", "6"]
    assert p["tally"] == [["5", 2], ["6", 1]]
    assert p["label"] == "5"
    assert p["ratio"] == 0.5
    assert p["rewards"] == [1.0, 1.0, 0.0, 0.0]
    assert p["truth_rewards"] == [1.0, 1.0, 0.0, 0.0]
    assert p["metrics"] == {
        "label_accuracy": 1.0,
        "reward_accuracy": 1.0,
        "majority_voting_reward": 0.5,
        "ground_truth_reward": 0.5,
        "pass@4": 1.0,
    }
    assert r["metrics"]["majority_ratio"] == 0.5


def check_extract_forms(solve):
    """抽取口径：嵌套花括号、最后一个 box、空 box、\fbox、无 box。"""
    texts = [
        "\\boxed{\\frac{1}{2}}",          # 嵌套花括号
        "first \\boxed{1} then \\boxed{2}",  # 最后一个生效
        "\\boxed{}",                       # 空 box 是合法票
        "\\boxed{1",                       # 花括号不配平
        "\\boxed 5 and {x}",               # 配平到了后面的花括号但前缀不对
        "\\fbox{7}",                       # 没有 \boxed 子串
        "answer: 7",                       # 完全没有 box
        "\\boxed{a} \\boxed",              # 最后一个框没写完，前面的不救场
        "\\boxed{1}}",                     # 多余的反括号被忽略
        "\\boxed}{7}",                     # 反括号先出现，计数变负
    ]
    r = _same(solve, texts, len(texts), ["anything"])
    assert r["prompts"][0]["extracted"] == [
        "\\frac{1}{2}", "2", "", None, None, None, None, None, "1", None,
    ]


def check_no_answer_at_all(solve):
    """一条都抽不出：标签是字符串 "None"、比例 0，且它照样参与判分。"""
    r = _same(solve, ["plain a", "plain b"], 2, ["None"])
    p = r["prompts"][0]
    assert p["label"] == "None"
    assert p["ratio"] == 0.0
    assert p["tally"] == []
    assert p["answers"] == []
    assert p["rewards"] == [0.0, 0.0]
    # 真值恰好也是 "None"：label_accuracy 给 1.0（"None" 就是个普通字符串）
    assert p["metrics"]["label_accuracy"] == 1.0
    assert p["metrics"]["pass@2"] == 0.0

    r = _same(solve, ["plain a", "plain b"], 2, ["5"])
    assert r["prompts"][0]["metrics"]["label_accuracy"] == 0.0


def check_none_label_is_votable(solve):
    """`\\boxed{None}` 抽出来的字符串能和兜底标签 "None" 撞上。"""
    r = _same(solve, ["plain text", "\\boxed{None}"], 2, ["None"])
    p = r["prompts"][0]
    assert p["extracted"] == [None, "None"]
    assert p["label"] == "None"
    assert p["ratio"] == 0.5
    assert p["rewards"] == [0.0, 1.0]
    assert p["truth_rewards"] == [0.0, 1.0]
    assert p["metrics"]["label_accuracy"] == 1.0
    assert p["metrics"]["reward_accuracy"] == 1.0


def check_ratio_denominator(solve):
    """比例的分母是样本数，不是成功抽出的条数。"""
    r = _same(solve, [_box("7"), "nope", "nope", "nope"], 4, ["7"])
    p = r["prompts"][0]
    assert p["ratio"] == 0.25
    assert p["metrics"]["majority_voting_reward"] == 0.25


def check_tie_first_seen(solve):
    """并列取首次出现者；票仓按首次出现序排列。"""
    r = _same(solve, [_box("2"), _box("3")], 2, ["3"])
    p = r["prompts"][0]
    assert p["tally"] == [["2", 1], ["3", 1]]
    assert p["label"] == "2"
    assert p["ratio"] == 0.5
    assert p["rewards"] == [1.0, 0.0]

    r = _same(solve, [_box("b"), _box("a"), _box("a"), _box("b")], 4, ["a"])
    p = r["prompts"][0]
    assert p["tally"] == [["b", 2], ["a", 2]]
    assert p["label"] == "b"


def check_canon_merge(solve):
    """化简表把不同写法并成一票；未命中原样保留。"""
    r = _same(solve, [_box("1/2"), _box("0.5"), _box("1/2")], 3, ["0.5"],
              canon={"1/2": "0.5"})
    p = r["prompts"][0]
    assert p["extracted"] == ["1/2", "0.5", "1/2"]
    assert p["answers"] == ["0.5", "0.5", "0.5"]
    assert p["tally"] == [["0.5", 3]]
    assert p["label"] == "0.5"
    assert p["ratio"] == 1.0
    assert p["rewards"] == [1.0, 1.0, 1.0]
    assert p["metrics"]["label_accuracy"] == 1.0


def check_canon_applied_once(solve):
    """每个原始值只归一化一次：真值自己带 box 也要先抽再化简。"""
    canon = {"a": "b"}
    r = _same(solve, [_box("a"), _box("b")], 2, ["\\boxed{a}"], canon=canon)
    p = r["prompts"][0]
    assert p["answers"] == ["b", "b"]
    assert p["label"] == "b"
    assert p["metrics"]["label_accuracy"] == 1.0
    assert p["truth_rewards"] == [1.0, 1.0]

    # 链式表（a->b, b->c）不会连锁：伪标签已经是化简后的值，不再过一遍表。
    chain = {"a": "b", "b": "c"}
    r = _same(solve, [_box("a"), _box("b")], 2, ["b"], canon=chain)
    p = r["prompts"][0]
    assert p["answers"] == ["b", "c"]
    assert p["tally"] == [["b", 1], ["c", 1]]
    assert p["label"] == "b"
    assert p["rewards"] == [1.0, 0.0]
    assert p["truth_rewards"] == [0.0, 1.0]
    assert p["metrics"]["label_accuracy"] == 0.0
    assert p["metrics"]["reward_accuracy"] == 0.0


def check_truth_malformed_box(solve):
    """真值 box 写坏了：抽不出来，连空答案也不给判中，但不许抛异常。"""
    r = _same(solve, [_box("5"), "\\boxed{}"], 2, ["\\boxed{5"])
    p = r["prompts"][0]
    assert p["extracted"] == ["5", ""]
    assert p["label"] == "5"
    assert p["metrics"]["label_accuracy"] == 0.0
    assert p["truth_rewards"] == [0.0, 0.0]
    assert p["metrics"]["pass@2"] == 0.0
    assert p["metrics"]["ground_truth_reward"] == 0.0


def check_rewards_split(solve):
    """多数票奖励看伪标签、真值奖励看真值：两组可以完全不同。"""
    r = _same(solve, [_box("9"), _box("9"), _box("9"), _box("5")],
              4, ["5"])
    p = r["prompts"][0]
    assert p["label"] == "9"
    assert p["ratio"] == 0.75
    assert p["rewards"] == [1.0, 1.0, 1.0, 0.0]
    assert p["truth_rewards"] == [0.0, 0.0, 0.0, 1.0]
    assert p["metrics"]["reward_accuracy"] == 0.0
    assert p["metrics"]["pass@4"] == 1.0


def check_reward_accuracy_fraction(solve):
    """reward_accuracy 数的是「逐样本奖励相等」的对数，不是均值相等。"""
    r = _same(solve, [_box("9"), _box("9"), _box("7"), _box("8")],
              4, ["5"])
    p = r["prompts"][0]
    assert p["label"] == "9"
    assert p["rewards"] == [1.0, 1.0, 0.0, 0.0]
    assert p["truth_rewards"] == [0.0, 0.0, 0.0, 0.0]
    assert p["metrics"]["reward_accuracy"] == 0.5
    assert p["metrics"]["majority_voting_reward"] == 0.5
    assert p["metrics"]["ground_truth_reward"] == 0.0
    assert p["metrics"]["pass@4"] == 0.0


def check_empty_answer_vote(solve):
    """`\\boxed{}` 抽出空串：空串也能当选，也能和空真值判中。"""
    r = _same(solve, ["\\boxed{}", "\\boxed{}", "no box", "no box"], 4, [""])
    p = r["prompts"][0]
    assert p["extracted"] == ["", "", None, None]
    assert p["label"] == ""
    assert p["ratio"] == 0.5
    assert p["rewards"] == [1.0, 1.0, 0.0, 0.0]
    assert p["truth_rewards"] == [1.0, 1.0, 0.0, 0.0]
    assert p["metrics"]["label_accuracy"] == 1.0


def check_multi_prompt_average(solve):
    """跨题取均值，键序取第一题；majority_ratio 单独平均。"""
    outputs = [_box("5"), _box("5"), _box("6"), _box("7"),
               _box("1"), _box("2"), _box("2"), _box("2")]
    r = _same(solve, outputs, 4, ["5", "2"])
    assert r["num_prompts"] == 2
    assert [p["label"] for p in r["prompts"]] == ["5", "2"]
    assert [p["ratio"] for p in r["prompts"]] == [0.5, 0.75]
    assert list(r["metrics"]) == ["label_accuracy", "reward_accuracy",
                                 "majority_voting_reward",
                                 "ground_truth_reward", "pass@4",
                                 "majority_ratio"]
    assert r["metrics"]["majority_ratio"] == 0.625
    assert r["metrics"]["label_accuracy"] == 1.0
    # 0.5 与 0.75 的均值
    assert r["metrics"]["majority_voting_reward"] == 0.625
    assert r["metrics"]["ground_truth_reward"] == 0.625
    assert r["metrics"]["pass@4"] == 1.0


def check_keep_downsample(solve):
    """降采样后整条流水线按新的 n（= keep）重跑，含比例分母。"""
    outputs = [_box("5"), _box("5"), _box("6"), _box("6"),
               _box("7"), _box("8"), _box("8"), _box("8")]
    r = _same(solve, outputs, 4, ["5", "8"], keep=2)
    assert r["selected"] == [0, 1, 4, 5]
    assert r["n"] == 2
    assert r["num_prompts"] == 2
    assert [p["extracted"] for p in r["prompts"]] == [["5", "5"], ["7", "8"]]
    assert [p["label"] for p in r["prompts"]] == ["5", "7"]
    assert [p["ratio"] for p in r["prompts"]] == [1.0, 0.5]
    assert list(r["metrics"])[-1] == "majority_ratio"
    assert r["metrics"]["majority_ratio"] == 0.75


def check_keep_equals_n(solve):
    """keep == n 时与不降采样完全一致。"""
    outputs = [_box("5"), _box("5"), _box("6"), "no box"]
    a = _same(solve, outputs, 4, ["5"], keep=4)
    b = _same(solve, outputs, 4, ["5"])
    assert a["selected"] == [0, 1, 2, 3]
    assert a["n"] == b["n"] == 4
    for key in ("prompts", "metrics", "num_prompts"):
        assert a[key] == b[key]


def check_keep_spills(solve):
    """keep > n 时官方不设防：下标吃到下一块，最后一块越界报 IndexError。"""
    outputs = [_box("1"), _box("2"), _box("3"), _box("4")]
    try:
        solve(outputs, 2, ["1", "3"], keep=3)
    except IndexError:
        pass
    else:
        raise AssertionError("expected IndexError for keep > n")


def check_bad_inputs(solve):
    """显式的入参校验一律 ValueError。"""
    cases = [
        ([_box("1")], 0, ["1"], None, None),
        ([_box("1"), _box("2")], 2, [], None, None),
        ([_box("1"), _box("2"), _box("3")], 2, ["1"], None, None),
        ([], 1, [], None, None),
        ([_box("1")], 1, ["1"], None, 0),
        ([_box("1")], 1, ["1"], None, -2),
        # 官方在降采样前先断言整批能被 n 整除；不整除时不许静悄悄吞掉尾巴
        ([_box("1"), _box("2"), _box("3"), _box("4"), _box("5")],
         2, ["1", "2"], None, 1),
    ]
    for outputs, n, gt, canon, keep in cases:
        try:
            solve(outputs, n, gt, canon=canon, keep=keep)
        except ValueError:
            continue
        raise AssertionError("expected ValueError for %r" % ((outputs, n, gt),
                                                             ))


check_example.example = True
