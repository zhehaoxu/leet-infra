import re


def _normalize_oracle(s):
    """独立实现的归一化 oracle（与题面 8 条规则一致，不 import solution）。"""
    s = s.strip()
    s = s.lower()
    # 千位逗号：手动扫描（与 regex 等价的独立写法）
    out = []
    for i, ch in enumerate(s):
        if ch == ',' and 0 < i < len(s) - 1 and s[i-1].isdigit() and s[i+1].isdigit():
            continue
        out.append(ch)
    s = ''.join(out)
    if s.endswith('.'):
        s = s[:-1]
    # 分数统一：先找所有 (a / b) 与 (a ÷ b) 的 span，再重建字符串
    spans = []
    i, n = 0, len(s)
    while i < n:
        if s[i] in '/÷':
            j = i - 1
            while j >= 0 and s[j] == ' ':
                j -= 1
            k = j
            while k >= 0 and s[k].isdigit():
                k -= 1
            left = s[k + 1:j + 1]
            m = i + 1
            while m < n and s[m] == ' ':
                m += 1
            e = m
            while e < n and s[e].isdigit():
                e += 1
            right = s[m:e]
            if left and right and (k < 0 or not s[k].isdigit()) and (e >= n or not s[e].isdigit()):
                spans.append((k + 1, e, left + '/' + right))
                i = e
                continue
        i += 1
    res = []
    pos = 0
    for a, b, rep in spans:
        res.append(s[pos:a])
        res.append(rep)
        pos = b
    res.append(s[pos:])
    s = ''.join(res)
    if s.endswith('%'):
        s = s[:-1]
    if s.startswith('$'):
        s = s[1:]
    return s


def _run(solve, samples, weights=None):
    return solve(samples, weights)


def check_normalization_rules(solve):
    cases = [
        ("  42  ", "42"),
        ("12,345", "12345"),
        ("1,2", "12"),             # 逗号前后都是数字 -> 删（规则 3 字面语义）
        ("42.", "42"),
        ("1 ÷ 2", "1/2"),
        ("7 / 8", "7/8"),
        ("12/5", "12/5"),
        ("42%", "42"),
        ("$99", "99"),
        ("2.5.", "2.5"),
        ("0.5", "0.5"),            # 不与 1/2 合并
        ("a / b", "a / b"),        # 非数字不套分数规则
        ("10 %", "10 "),           # 末尾 % 一律剥离（含空格隔开），残留尾空格
    ]
    for raw, exp in cases:
        assert _normalize_oracle(raw) == exp, (raw, _normalize_oracle(raw), exp)


def check_basic_vote(solve):
    samples = ["42", " 42 ", "43", "42.", "$42"]
    led = solve(samples)
    assert led["normalized"] == ["42", "42", "43", "42", "42"]
    assert led["tally"] == {"42": 4.0, "43": 1.0}
    assert led["winner"] == "42"
    assert abs(led["vote_share"] - 0.8) < 1e-9
    assert led["n_candidates"] == 2


def check_fraction_merge(solve):
    samples = ["1/2", "1 ÷ 2", "0.5"]
    led = solve(samples)
    assert led["tally"] == {"1/2": 2.0, "0.5": 1.0}
    assert led["winner"] == "1/2"


def check_weighted_vote(solve):
    samples = ["a", "b", "b", "a"]
    led = solve(samples, weights=[3.0, 1.0, 1.0, 1.0])
    assert led["tally"] == {"a": 4.0, "b": 2.0}
    assert led["winner"] == "a"
    assert abs(led["vote_share"] - 4.0 / 6.0) < 1e-9


def check_tie_first_occurrence(solve):
    # 票重并列时取「首次出现」者：dict 保序 + 严格大于才覆盖
    led = solve(["b", "a", "b", "a"])
    assert led["tally"] == {"b": 2.0, "a": 2.0}
    assert led["winner"] == "b"
    led2 = solve(["a", "b", "b", "a"])
    assert led2["winner"] == "a"
    # 加权并列
    led3 = solve(["x", "y"], weights=[2.5, 2.5])
    assert led3["winner"] == "x"


def check_case_and_percent_via_solve(solve):
    # 大小写合并与 % 剥离必须经 solve 的真实归一化路径验证
    led = solve(["OK", "ok", "Ok"])
    assert led["tally"] == {"ok": 3.0}
    led2 = solve(["42%", "42"])
    assert led2["tally"] == {"42": 2.0}
    # 规则边界：分数规则只作用于纯整数操作数，字母保持原样不合并
    led3 = solve(["A/2", "a ÷ 2"])
    assert led3["tally"] == {"a/2": 1.0, "a ÷ 2": 1.0}


def check_empty_and_zero_weight(solve):
    led = solve([])
    assert led["normalized"] == [] and led["tally"] == {}
    assert led["winner"] is None and led["vote_share"] == 0.0
    assert led["n_candidates"] == 0
    # 全零票重：share 取 0.0，winner 仍取首次出现
    led2 = solve(["p", "q"], weights=[0.0, 0.0])
    assert led2["vote_share"] == 0.0
    assert led2["winner"] == "p"


def check_order_preserved_in_tally(solve):
    led = solve(["z", "y", "z", "x", "y", "z"])
    assert list(led["tally"].keys()) == ["z", "y", "x"]
    assert led["tally"]["z"] == 3.0


def check_negative_weight_rejected(solve):
    try:
        solve(["a"], weights=[-1.0])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for negative weight")
    try:
        solve(["a", "b"], weights=[1.0])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for length mismatch")


def check_long_sample_realistic(solve):
    samples = ["1,234.", "1234", "1/2", "1 ÷ 2", "$1,234"]  # 调用方已截取答案段
    led = solve(samples)
    assert led["tally"] == {"1234": 3.0, "1/2": 2.0}
    assert abs(led["vote_share"] - 0.6) < 1e-9
