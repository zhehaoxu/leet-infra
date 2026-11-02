import re

_OP_RE = re.compile(r"((?:REMOVE|EDIT|ADD|AGREE)(?: \d+|)): (?:[a-zA-Z\s\d]+: |)(.*)")
_BANNED = ("ADD", "AGREE", "EDIT")


def _eq(got, ref, path=""):
    if isinstance(ref, dict):
        assert isinstance(got, dict), path
        assert set(got) == set(ref), (path, sorted(got), sorted(ref))
        for k in ref:
            _eq(got[k], ref[k], path + "/" + str(k))
        return
    if isinstance(ref, (list, tuple)):
        assert isinstance(got, (list, tuple)), (path, type(got))
        assert len(got) == len(ref), (path, len(got), len(ref))
        for i, (g, r) in enumerate(zip(got, ref)):
            _eq(g, r, path + "[%d]" % i)
        return
    assert got == ref, (path, got, ref)


def _parse(text):
    res = []
    for operation, body in _OP_RE.findall(text):
        body = body.strip()
        if body == "" or any(w in body for w in _BANNED) or not body.endswith("."):
            continue
        res.append(["ADD", body] if "ADD" in operation else [operation.strip(), body])
    return res


def _find(rules, text):
    for i in range(len(rules)):
        if rules[i][0] in text:
            return i
    return None


def _update(rules, ops, full):
    dead = []
    for i in range(len(ops)):
        op, text = ops[i]
        kind = op.split(" ")[0]
        num = int(op.split(" ")[1]) if " " in op else None
        if kind == "ADD":
            if _find(rules, text) is not None:
                dead.append(i)
        elif kind == "EDIT":
            j = _find(rules, text)
            if j is not None:
                ops[i] = ["AGREE %d" % (j + 1), rules[j][0]]
            elif num is None or num > len(rules):
                dead.append(i)
        elif kind in ("REMOVE", "AGREE"):
            if _find(rules, text) is None:
                dead.append(i)
        else:
            raise AssertionError("未知操作 %r" % (kind,))
    live = [ops[i] for i in range(len(ops)) if i not in dead]
    for wanted in ("REMOVE", "AGREE", "EDIT", "ADD"):
        for op, text in live:
            if op.split(" ")[0] != wanted:
                continue
            if wanted == "REMOVE":
                rules[_find(rules, text)][1] -= 3 if full else 1
            elif wanted == "AGREE":
                rules[_find(rules, text)][1] += 1
            elif wanted == "EDIT":
                k = int(op.split(" ")[1]) - 1
                rules[k] = [text, rules[k][1] + 1]
            else:
                rules.append([text, 2])
    rules = [r for r in rules if r[1] > 0]
    rules.sort(key=lambda r: -r[1])
    return rules, live


def _py(rounds, initial_rules=None, max_num_rules=10):
    rules = [[t, c] for t, c in (initial_rules or ())]
    log = []
    for i, raw in enumerate(rounds):
        ops = _parse(raw)
        parsed = [list(o) for o in ops]
        full = max_num_rules + 5 <= len(rules)
        rules, live = _update(rules, ops, full)
        log.append({"round": i, "parsed": parsed,
                    "applied": [list(o) for o in live],
                    "list_full": full, "rules": [list(r) for r in rules]})
    prompt = "\n".join("%d. %s" % (i, r[0]) for i, r in enumerate(rules, 1))
    return {"rules": rules, "prompt": prompt, "log": log, "count": len(rules)}


def _run(solve, rounds, **kw):
    ref = _py(list(rounds), **{k: (v if k != "initial_rules" else [list(x) for x in v])
                               for k, v in kw.items()})
    got = solve(list(rounds), **{k: (v if k != "initial_rules" else [list(x) for x in v])
                                 for k, v in kw.items()})
    _eq(got, ref)
    return got


def check_add_parses_into_count_two(solve):
    """ADD 的初始计数是 2（不是 1），并且编号 `ADD 3:` 里的数字被丢掉。"""
    led = _run(solve, ["ADD: Always check the drawer.",
                       "ADD 7: Always close the fridge."])
    assert led["rules"] == [["Always check the drawer.", 2],
                            ["Always close the fridge.", 2]]
    assert led["log"][1]["parsed"] == [["ADD", "Always close the fridge."]]


def check_parse_prefix_label(solve):
    """`ADD: Label: 正文` 里的标签前缀被吞掉；标签若把整行吃掉就只剩残句而被拒。"""
    led = _run(solve, ["ADD: Rule: Always check the drawer.",
                       "ADD: Note: Always\ncheck the box."])
    assert led["log"][0]["parsed"] == [["ADD", "Always check the drawer."]]
    assert led["log"][1]["parsed"] == []      # 只剩 "Always"，没有句号


def check_parse_three_gates(solve):
    """三道门：空串、不以句号结尾、正文里出现 ADD/AGREE/EDIT 一律丢弃；REMOVE 不违规。"""
    text = "\n".join([
        "ADD: ",
        "ADD: no period",
        "ADD: do not ADD duplicates.",
        "ADD: always AGREE first.",
        "ADD: EDIT the file in place.",
        "ADD: do a REMOVE pass last.",
        "EDIT 2: keep the REMOVE word.",
    ])
    led = _run(solve, [text])
    assert [op[1] for op in led["log"][0]["parsed"]] == [
        "do a REMOVE pass last.",
        "keep the REMOVE word.",
    ]


def check_parse_midword_and_same_line(solve):
    """正则不锚定：`PADD:` 也命中；同一行写两条操作会**双双**丢掉（第一条吃掉整行）。"""
    led = _run(solve, ["PADD: word boundary is not checked."])
    assert led["log"][0]["parsed"] == [["ADD", "word boundary is not checked."]]
    led = _run(solve, ["ADD: first rule. ADD: second rule."])
    assert led["log"][0]["parsed"] == []
    assert led["count"] == 0 and led["prompt"] == ""


def check_add_dedupes_by_substring(solve):
    """ADD 去重是「已有规则文本是新文本的子串」才算重复，方向反了就不算。"""
    led = _run(solve, ["ADD: Check the drawer. Then open it."],
               initial_rules=[["Check the drawer.", 5]])
    assert led["rules"] == [["Check the drawer.", 5]]
    assert led["log"][0]["applied"] == []

    led = _run(solve, ["ADD: Check the drawer."],
               initial_rules=[["Check the drawer. Then open it.", 5]])
    assert led["rules"] == [["Check the drawer. Then open it.", 5],
                            ["Check the drawer.", 2]]


def check_duplicate_add_in_one_batch(solve):
    """同一批里两条一模一样的 ADD 都留：去重只看批前的旧规则。"""
    led = _run(solve, ["ADD: Check the drawer.\nADD: Check the drawer."])
    assert led["rules"] == [["Check the drawer.", 2], ["Check the drawer.", 2]]
    assert led["prompt"] == "1. Check the drawer.\n2. Check the drawer."


def check_edit_becomes_agree(solve):
    """EDIT 的新文本里含已有规则时，改判成 AGREE：文本不动、计数 +1、序号取**首个**匹配。"""
    led = _run(solve, ["EDIT 4: Check the drawer. Then open it."],
               initial_rules=[["Check the drawer.", 5], ["Wipe the shelf.", 1]])
    assert led["log"][0]["parsed"] == [["EDIT 4", "Check the drawer. Then open it."]]
    assert led["log"][0]["applied"] == [["AGREE 1", "Check the drawer."]]
    assert led["rules"] == [["Check the drawer.", 6], ["Wipe the shelf.", 1]]


def check_edit_by_number(solve):
    """EDIT 按**编号**定位（不是按文本），换文本并把计数 +1。"""
    led = _run(solve, ["EDIT 2: Always wear gloves."],
               initial_rules=[["Check the drawer.", 1], ["Open the box.", 4]])
    assert led["rules"] == [["Always wear gloves.", 5], ["Check the drawer.", 1]]


def check_edit_invalid_number(solve):
    """EDIT 没编号、或编号超过当前条数，整条丢弃。"""
    led = _run(solve, ["EDIT: Always wear gloves.\nEDIT 9: Always wear boots."],
               initial_rules=[["Check the drawer.", 1]])
    assert led["log"][0]["applied"] == []
    assert led["rules"] == [["Check the drawer.", 1]]


def check_remove_strength(solve):
    """REMOVE 平时 -1；列表长度够 `max_num_rules + 5` 时一次 -3。"""
    base = [["r1.", 4], ["r2.", 2], ["r3.", 6], ["r4.", 1]]
    led = _run(solve, ["REMOVE 3: r3."], initial_rules=base)
    assert led["log"][0]["list_full"] is False
    assert ["r3.", 5] in led["rules"]

    full = base + [["r5.", 3]]
    led = _run(solve, ["REMOVE 3: r3."], initial_rules=full, max_num_rules=0)
    assert led["log"][0]["list_full"] is True
    assert ["r3.", 3] in led["rules"]


def check_remove_unknown_text(solve):
    """REMOVE / AGREE 的文本匹配不到任何已有规则时整条丢弃。"""
    led = _run(solve, ["REMOVE 1: never seen.\nAGREE: never seen either."],
               initial_rules=[["Check the drawer.", 3]])
    assert led["log"][0]["applied"] == []
    assert led["rules"] == [["Check the drawer.", 3]]


def check_prune_and_stable_sort(solve):
    """计数降到 0 的规则被剔除；存活规则按计数降序、同分保持原有先后。"""
    led = _run(solve, ["REMOVE 1: r1.\nAGREE: r1b."],
               initial_rules=[["r1.", 1], ["r1b.", 2], ["r3.", 2]])
    assert led["rules"] == [["r1b.", 3], ["r3.", 2]]

    led = _run(solve, ["AGREE: r1."],
               initial_rules=[["r1.", 1], ["r2.", 2], ["r3.", 2]])
    assert [r[0] for r in led["rules"]] == ["r1.", "r2.", "r3."]
    assert [r[1] for r in led["rules"]] == [2, 2, 2]


def check_apply_order_agree_before_edit(solve):
    """同一批里 AGREE 先按文本落地、EDIT 再按编号换文本：换序会让 AGREE 找不到原文。"""
    led = _run(solve, ["AGREE: Check the drawer.\nEDIT 1: Always wear gloves."],
               initial_rules=[["Check the drawer.", 1]])
    assert led["rules"] == [["Always wear gloves.", 3]]


def check_round_snapshots(solve):
    """每轮的 `list_full` 取的是**该轮开始前**的长度；ADD 之后长度变化要到下一轮才生效。"""
    base = [["r%d." % i, 5] for i in range(1, 5)]      # 4 条，max=0 -> 5<=4 False
    led = _run(solve, ["ADD: a new rule.", "REMOVE 1: r1."],
               initial_rules=base, max_num_rules=0)
    assert [rec["list_full"] for rec in led["log"]] == [False, True]
    assert led["log"][1]["rules"] == [["r2.", 5], ["r3.", 5], ["r4.", 5],
                                      ["r1.", 2], ["a new rule.", 2]]


def check_render_prompt(solve):
    """渲染是 `"<序号>. <文本>"` 逐行、序号从 1 起、没有尾换行；空账渲染成空串。"""
    led = _run(solve, ["ADD: b rule.\nADD: a rule."])
    assert led["prompt"] == "1. b rule.\n2. a rule."
    assert not led["prompt"].endswith("\n")
    empty = _run(solve, [])
    assert empty == {"rules": [], "prompt": "", "log": [], "count": 0}


def check_validation(solve):
    """结构校验：rounds 非序列 / 元素非字符串、max_num_rules 非整数或为负、初值形状错误。"""
    for args in ((None,), ("ADD: x.",), ([b"ADD: x."],), ([["ADD: x."]],)):
        try:
            solve(*args)
        except ValueError:
            pass
        else:
            raise AssertionError("%r 应当报错" % (args,))
    for kw in ({"max_num_rules": -1}, {"max_num_rules": True},
               {"max_num_rules": 2.0}, {"max_num_rules": "3"}):
        try:
            solve([], **kw)
        except ValueError:
            pass
        else:
            raise AssertionError("%r 应当报错" % (kw,))
    for bad in ([[["rule."]]], [[["rule.", 1, 2]]], [[[1, 2]]], [[["rule.", True]]]):
        try:
            solve([], initial_rules=bad)
        except ValueError:
            pass
        else:
            raise AssertionError("%r 应当报错" % (bad,))


def check_against_oracle(solve):
    """混合多轮：新增 / 重复新增 / 改判 AGREE / 按编号改写 / 满额删除 / 自然淘汰。"""
    rounds = [
        "ADD: Check the drawer.\nADD: Wipe the shelf.",
        "ADD: Check the drawer. Then open it.\nEDIT 3: Wipe the shelf first.",
        "AGREE: Check the drawer. Then open it. And close.",
        "REMOVE 1: Check the drawer.\nEDIT 2: Always wear gloves.",
        "AGREE: Always wear gloves.\nADD: Always close the fridge.",
    ]
    led = _run(solve, rounds, max_num_rules=1)
    assert [r[0] for r in led["rules"]] == [
        "Always wear gloves.", "Check the drawer.", "Always close the fridge."]
    assert [r[1] for r in led["rules"]] == [4, 2, 2]
    # 第二轮整批空转：ADD 的正文含旧规则（被去重），EDIT 3 的编号越界
    assert led["log"][1]["applied"] == []
    assert [rec["list_full"] for rec in led["log"]] == [False] * 5


def run_examples(solve):
    print(_run(solve, ["ADD: Check the drawer.", "AGREE: Check the drawer."]))
    print(_run(solve, ["EDIT 1: Check the drawer. Then open it."],
               initial_rules=[["Check the drawer.", 5]])["log"][0]["applied"])


check_add_parses_into_count_two.example = True
check_remove_strength.example = True
