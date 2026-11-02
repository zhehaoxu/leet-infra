import unicodedata


# ---------------------------------------------------------------- oracle
# 独立路径：不用 re，手写扫描器复刻 '\d+\.?\d*' 与 ',\s?' 的语义；均值表也重写一遍。

def _is_digit(ch):
    return unicodedata.category(ch) == "Nd"


def _is_space(ch):
    return ch in " \t\n\r\f\v" or unicodedata.category(ch) in ("Zs", "Zl", "Zp")


def _num_at(s, i):
    """枚举 s[i:] 对 \\d+\\.?\\d* 的候选（正则的回溯序）。"""
    n, j = len(s), i
    while j < n and _is_digit(s[j]):
        j += 1
    if j == i:
        return []
    out = []
    if j < n and s[j] == ".":
        k = j + 1
        while k < n and _is_digit(s[k]):
            k += 1
        out.append((s[i:k], k))
    out.append((s[i:j], j))
    return out


def _scan_pair(s, double):
    n = len(s)
    for i in range(n):
        if double:
            if s[i:i + 2] != "[[":
                continue
            start = i + 2
        else:
            if s[i] != "[":
                continue
            start = i + 1
        for tok1, j in _num_at(s, start):
            if j >= n or s[j] != ",":
                continue
            p = j + 1
            starts = [p + 1, p] if (p < n and _is_space(s[p])) else [p]
            for st in starts:
                for tok2, k in _num_at(s, st):
                    tail = "]]" if double else "]"
                    if s[k:k + len(tail)] == tail:
                        return tok1, tok2
    return None


def _scan_one(s, double):
    n = len(s)
    open_len = 2 if double else 1
    tail = "]]" if double else "]"
    for i in range(n):
        if s[i:i + open_len] != "[" * open_len:
            continue
        for tok, j in _num_at(s, i + open_len):
            if s[j:j + len(tail)] == tail:
                return tok
    return None


def _num(tok):
    if "." in tok:
        return float(tok)
    if len(tok) > 1 and tok[0] == "0":
        raise ValueError(f"非法分数字面量: {tok!r}")
    return int(tok)


def _pair(text):
    got = _scan_pair(text, True)
    if got is not None:
        return "two", [_num(got[0]), _num(got[1])]
    got = _scan_pair(text, False)
    if got is not None:
        return "two_backup", [_num(got[0]), _num(got[1])]
    return "none", None


def _one(text):
    got = _scan_one(text, True)
    if got is not None:
        return "one", _num(got)
    got = _scan_one(text, False)
    if got is not None:
        return "one_backup", _num(got)
    return "none", None


def _py(entries, mode="pair", tie_delta=0.1):
    if mode not in ("pair", "single"):
        raise ValueError("mode")
    if isinstance(tie_delta, bool) or not isinstance(tie_delta, (int, float)):
        raise ValueError("tie_delta")
    if tie_delta != tie_delta or tie_delta < 0 or tie_delta == float("inf"):
        raise ValueError("tie_delta")
    if not isinstance(entries, (list, tuple)):
        raise ValueError("entries")

    if mode == "single":
        rows, per = [], {}
        for e in entries:
            if set(("model", "turn", "text")) - set(e):
                raise ValueError("keys")
            if not isinstance(e["model"], str) or not e["model"]:
                raise ValueError("model")
            if not isinstance(e["turn"], int) or isinstance(e["turn"], bool):
                raise ValueError("turn")
            if e["turn"] not in (1, 2):
                raise ValueError("turn")
            if not isinstance(e["text"], str):
                raise ValueError("text")
            pat, sc = _one(e["text"])
            rows.append({"model": e["model"], "turn": e["turn"],
                         "pattern": pat, "score": -1 if sc is None else sc})
        for r in rows:
            if r["score"] == -1:
                continue
            per.setdefault(r["model"], {1: [], 2: []})[r["turn"]].append(r["score"])
        table = {}
        for name, b in per.items():
            both = b[1] + b[2]
            table[name] = {
                "turn_1": (sum(b[1]) / len(b[1])) if b[1] else None,
                "turn_2": (sum(b[2]) / len(b[2])) if b[2] else None,
                "avg": sum(both) / len(both),
            }
        order = sorted(table, key=lambda n: (-table[n]["avg"], n))
        return {"mode": mode, "rows": rows, "table": table, "rank": order}

    rows = []
    for text in entries:
        if not isinstance(text, str):
            raise ValueError("text")
        pat, sc = _pair(text)
        if sc is None:
            verdict = "error"
        elif abs(sc[0] - sc[1]) <= tie_delta:
            verdict = "tie"
        else:
            verdict = "A" if sc[0] > sc[1] else "B"
        rows.append({"pattern": pat, "scores": sc, "verdict": verdict})
    return {"mode": mode, "rows": rows, "table": {}, "rank": []}


# ---------------------------------------------------------------- compare

def _eq(a, b):
    if isinstance(a, float) or isinstance(b, float):
        if a is None or b is None:
            return a is b
        return abs(a - b) <= 1e-12
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_eq(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_eq(x, y) for x, y in zip(a, b))
    return a == b


def _compare(solve, entries, **kw):
    led = solve(list(entries), **kw)
    ref = _py(entries, **kw)
    assert _eq(led, ref), f"\nled={led}\nref={ref}"
    return led


# ---------------------------------------------------------------- 成对打分

def check_pair_basic(solve):
    led = _compare(solve, ["[[8, 7]]", "[[7, 8]]", "[[9, 9]]"])
    r = led["rows"]
    assert [x["pattern"] for x in r] == ["two", "two", "two"]
    assert r[0]["scores"] == [8, 7] and r[0]["verdict"] == "A"
    assert r[1]["scores"] == [7, 8] and r[1]["verdict"] == "B"
    assert r[2]["verdict"] == "tie"
    assert led["table"] == {} and led["rank"] == []


def check_pair_whitespace(solve):
    """'\\s?' 恰一个 Unicode 空白：零个 / 一个（含 tab、换行、NBSP、全角空格）都命中，两个不命中。"""
    hits = ["[[8,7]]", "[[8, 7]]", "[[8,\t7]]", "[[8,\n7]]",
            "[[8,\xa07]]", "[[8,\u30007]]", "[[8,\r7]]", "[[8,\f7]]"]
    for text in hits:
        led = _compare(solve, [text])
        assert led["rows"][0]["pattern"] == "two", text
        assert led["rows"][0]["scores"] == [8, 7], text
    misses = ["[[8,  7]]", "[[8,\t\t7]]", "[[8, \t7]]"]
    for text in misses:
        led = _compare(solve, [text])
        assert led["rows"][0]["pattern"] == "none", text
        assert led["rows"][0]["scores"] is None
        assert led["rows"][0]["verdict"] == "error"


def check_pair_bracket(solve):
    """`[[` 后必须紧跟数字；单层括号走备份模式；备份模式取最左内层；无尾锚。"""
    led = _compare(solve, ["[[ 8, 7]]", "[1, 2]", "[[8,7],[9,2]]",
                           "[[8,7]] and [[9,2]]", "[[8, 7]]x", "[[8, 7]x"])
    p = [x["pattern"] for x in led["rows"]]
    assert p == ["none", "two_backup", "two_backup", "two", "two", "two_backup"], p
    assert led["rows"][2]["scores"] == [8, 7]   # 备份模式取最左内层
    assert led["rows"][3]["scores"] == [8, 7]   # 主模式取最左完整块
    assert led["rows"][0]["verdict"] == "error"
    # 负数 / 只有一半 / 纯文本：都不中
    for text in ["[[-1, 7]]", "[[8, 7", "no score here", ""]:
        led = _compare(solve, [text])
        assert led["rows"][0]["pattern"] == "none", text


def check_pair_tie_delta(solve):
    """浮点边界：abs(8.0-8.1)=0.0999…<=0.1 平；abs(1.0-1.1)=0.1000…009>0.1 不平。"""
    led = _compare(solve, ["[[8.0, 8.1]]", "[[1.0, 1.1]]", "[[7, 7]]"])
    v = [x["verdict"] for x in led["rows"]]
    assert v == ["tie", "B", "tie"], v

    led = _compare(solve, ["[[8.0, 8.1]]", "[[7, 7]]", "[[7, 8]]"], tie_delta=0)
    v = [x["verdict"] for x in led["rows"]]
    assert v == ["B", "tie", "B"], v

    led = _compare(solve, ["[[5.5, 5.55]]", "[[5.5, 5.56]]"], tie_delta=0.05)
    assert led["rows"][0]["verdict"] == "tie"  # 名义差 0.05，浮点落在 0.049999…982 之下
    assert led["rows"][1]["verdict"] == "B"


def check_number_kinds(solve):
    """无小数点得 int、有得 float；`5.` 合法；前导零抛 ValueError。"""
    led = _compare(solve, ["[[8, 7]]", "[[8.0, 7]]", "[[8., 7.]]"])
    r = led["rows"]
    assert isinstance(r[0]["scores"][0], int)
    assert isinstance(r[1]["scores"][0], float)
    assert r[2]["scores"] == [8.0, 7.0]
    for text in ["[[007, 7]]", "[[8, 08]]"]:
        try:
            _compare(solve, [text])
        except ValueError:
            pass
        else:
            raise AssertionError(f"{text} 应抛 ValueError")


# ---------------------------------------------------------------- 单点评分

def check_single_parse(solve):
    rows = [
        {"model": "a", "turn": 1, "text": "rating: [[8]]"},
        {"model": "a", "turn": 1, "text": "[6]"},
        {"model": "a", "turn": 1, "text": "[[9.]]"},
        {"model": "a", "turn": 1, "text": "[[8, 7]]"},
        {"model": "a", "turn": 1, "text": "[[-1]]"},
        {"model": "a", "turn": 1, "text": ""},
        {"model": "c", "turn": 1, "text": "[[0]]"},
    ]
    led = _compare(solve, rows, mode="single")
    r = led["rows"]
    assert [x["pattern"] for x in r] == [
        "one", "one_backup", "one", "none", "none", "none", "one"]
    assert [x["score"] for x in r] == [8, 6, 9.0, -1, -1, -1, 0]
    assert led["table"]["c"]["avg"] == 0.0  # 0 是合法分数，不能当 -1 哨兵


def check_single_missing_is_minus_one(solve):
    """单点模式喂成对格式 / 负数 / 空串，一律 -1 并在聚合时剔除；全 -1 的模型整行消失。"""
    rows = [
        {"model": "a", "turn": 1, "text": "[[8, 7]]"},
        {"model": "b", "turn": 1, "text": "[[5]]"},
    ]
    led = _compare(solve, rows, mode="single")
    assert "a" not in led["table"] and led["rank"] == ["b"]
    assert led["table"]["b"]["turn_1"] == 5.0
    assert led["table"]["b"]["turn_2"] is None


def check_single_table(solve):
    rows = [
        {"model": "a", "turn": 1, "text": "[[10]]"},
        {"model": "a", "turn": 1, "text": "bad"},
        {"model": "a", "turn": 1, "text": "[[8]]"},
        {"model": "a", "turn": 2, "text": "[[4]]"},
        {"model": "b", "turn": 2, "text": "[[7]]"},
        {"model": "b", "turn": 1, "text": "[[3]]"},
    ]
    led = _compare(solve, rows, mode="single")
    a, b = led["table"]["a"], led["table"]["b"]
    assert abs(a["turn_1"] - 9.0) < 1e-12      # (10+8)/2，失败行不进均值
    assert a["turn_2"] == 4.0
    assert abs(a["avg"] - 22 / 3) < 1e-12
    assert b["turn_1"] == 3.0 and b["turn_2"] == 7.0
    assert abs(b["avg"] - 5.0) < 1e-12
    assert led["rank"] == ["a", "b"]           # 7.33 > 5.0


def check_single_rank(solve):
    """均值降序、同分按模型名升序（不能靠字典插入序）。"""
    rows = [
        {"model": "z", "turn": 1, "text": "[[6]]"},
        {"model": "y", "turn": 1, "text": "[[6]]"},
        {"model": "x", "turn": 2, "text": "[[6]]"},
        {"model": "w", "turn": 1, "text": "[[9]]"},
    ]
    led = _compare(solve, rows, mode="single")
    assert led["rank"] == ["w", "x", "y", "z"]


def check_empty(solve):
    led = _compare(solve, [])
    assert led["rows"] == [] and led["table"] == {} and led["rank"] == []
    led = _compare(solve, [], mode="single")
    assert led["rows"] == [] and led["table"] == {} and led["rank"] == []


# ---------------------------------------------------------------- 契约

def check_rejections(solve):
    bad = [
        lambda: solve([], mode="pairwise"),
        lambda: solve([], mode=1),
        lambda: solve([], tie_delta=True),
        lambda: solve([], tie_delta="0.1"),
        lambda: solve([], tie_delta=-0.1),
        lambda: solve([], tie_delta=float("nan")),
        lambda: solve([], tie_delta=float("inf")),
        lambda: solve("not a list"),
        lambda: solve([5]),
        lambda: solve([None]),
        lambda: solve([{"text": "[[1]]"}], mode="single"),
        lambda: solve([{"model": "a", "text": "[[1]]"}], mode="single"),
        lambda: solve([{"model": "a", "turn": 1}], mode="single"),
        lambda: solve([{"model": "", "turn": 1, "text": "[[1]]"}], mode="single"),
        lambda: solve([{"model": "a", "turn": 3, "text": "[[1]]"}], mode="single"),
        lambda: solve([{"model": "a", "turn": 0, "text": "[[1]]"}], mode="single"),
        lambda: solve([{"model": "a", "turn": True, "text": "[[1]]"}], mode="single"),
        lambda: solve([{"model": "a", "turn": 1.0, "text": "[[1]]"}], mode="single"),
        lambda: solve([{"model": "a", "turn": 1, "text": b"[[1]]"}], mode="single"),
        lambda: solve([{"model": "a", "turn": 1, "text": "[[007]]"}], mode="single"),
    ]
    for fn in bad:
        try:
            fn()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")


def check_contract(solve):
    """字段齐全、两模式互不污染、入参不被就地改写。"""
    pair = solve(["[[9, 2]]"])
    assert set(pair) == {"mode", "rows", "table", "rank"}
    assert pair["mode"] == "pair" and pair["table"] == {} and pair["rank"] == []
    assert set(pair["rows"][0]) == {"pattern", "scores", "verdict"}
    assert pair["rows"][0]["verdict"] == "A"

    single = solve([{"model": "a", "turn": 1, "text": "[[8]]"}], mode="single")
    assert single["mode"] == "single" and single["table"] != {}
    assert set(single["rows"][0]) == {"model", "turn", "pattern", "score"}

    rows = [{"model": "a", "turn": 1, "text": "[[8]]"}]
    snapshot = {k: v for k, v in rows[0].items()}
    solve(rows, mode="single")
    assert rows[0] == snapshot

    # tuple 也是合法容器
    assert solve(("[[9, 2]]", ""))["rows"][1]["verdict"] == "error"


check_pair_basic.example = True
check_single_table.example = True
