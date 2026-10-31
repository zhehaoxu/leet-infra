import hashlib
import re

_END_MARKS = (".", "?", "!", '"')
_ELLIPSIS = "..."
_POLICY = ["terms of use", "privacy policy", "cookie policy",
           "uses cookies", "use of cookies", "use cookies"]


def _py(text, badwords=None, seen_hashes=None,
        min_words_per_line=5, min_num_sentences=3, max_word_length=1000):
    seen = set() if seen_hashes is None else set(seen_hashes)
    bw_re = None
    if badwords:
        bw_re = re.compile("[^a-z]({})[^a-z]".format("|".join(badwords)))
    counters = dict(too_long=0, no_endmark=0, too_short=0, javascript=0,
                    policy=0, dedup=0)
    key_map = {"too_long": "lines_with_too_long_word", "no_endmark": "lines_no_endmark",
               "too_short": "lines_too_short", "javascript": "lines_javascript",
               "policy": "lines_policy", "dedup": "lines_dedup"}
    valid = []
    new_hashes = []
    reason = None

    def cit_sub(s):
        out = re.sub(r"\[\d*\]", "", s)
        out = out.replace("[edit]", "")
        return out.replace("[citation needed]", "")

    for raw in text.splitlines():
        line = raw.strip()
        long_word = False
        for w in line.split():
            if len(w) > max_word_length:
                long_word = True
                break
        if long_word:
            counters["too_long"] += 1
            continue
        line = cit_sub(line)
        has_end = False
        for m in _END_MARKS:
            if line.endswith(m):
                has_end = True
                break
        if (not has_end) or line.endswith(_ELLIPSIS):
            counters["no_endmark"] += 1
            continue
        if len(line.split()) < min_words_per_line:
            counters["too_short"] += 1
            continue
        low = line.lower()
        if "lorem ipsum" in low:
            reason = "filtered_page_loremipsum"
            break
        if "javascript" in low:
            counters["javascript"] += 1
            continue
        if "{" in line:
            reason = "filtered_page_squigglybracket"
            break
        hit_policy = False
        for p in _POLICY:
            if p in low:
                hit_policy = True
                break
        if hit_policy:
            counters["policy"] += 1
            continue
        if bw_re is not None and bw_re.search(low):
            reason = "filtered_page_badword"
            break
        h = hashlib.md5(line.strip().lower().encode("utf-8")).hexdigest()
        if h in seen:
            counters["dedup"] += 1
            continue
        seen.add(h)
        new_hashes.append(h)
        valid.append(line)

    if reason is None and len(valid) < min_num_sentences:
        reason = "filtered_page_toofewsentences"
    ok = reason is None
    ns = len(valid)
    return {
        "accepted": ok,
        "reject_reason": reason,
        "valid_lines": list(valid) if ok else [],
        "clean_text": "\n".join(valid).strip() if ok else "",
        "num_sentences": ns,
        "drop_counters": {key_map[k]: v for k, v in counters.items()},
        "new_hashes": list(new_hashes) if ok else [],
    }


def _compare(solve, text, badwords=None, seen_hashes=None,
             min_words_per_line=5, min_num_sentences=3, max_word_length=1000):
    led = solve(text, badwords, seen_hashes, min_words_per_line,
                min_num_sentences, max_word_length)
    ref = _py(text, badwords, seen_hashes, min_words_per_line,
              min_num_sentences, max_word_length)
    assert led == ref, f"\nled={led}\nref={ref}"
    return led


_W = " ".join(["word"] * 4)  # 4 词，不满足 5 词下限


def check_line_filters(solve):
    text = "\n".join([
        "The model was trained on data.",   # OK（5 词）
        _W + ".",                            # 4 词 → too_short
        "no endmark here",                   # 无终止符 → no_endmark
        "Trailing dots are bad...",          # "..." 结尾 → no_endmark
        "Please enable javascript here right now.",  # javascript → 丢行
        "We value your privacy policy.",     # policy → 丢行
        "A " + "x" * 1001 + " happened.",    # 超长词 → too_long_word
        "Another good sentence follows right here.",    # OK
        "A third fine line survives.",        # OK
    ])
    led = _compare(solve, text)
    c = led["drop_counters"]
    assert c["lines_too_short"] == 1
    assert c["lines_no_endmark"] == 2
    assert c["lines_javascript"] == 1
    assert c["lines_policy"] == 1
    assert c["lines_with_too_long_word"] == 1
    assert c["lines_dedup"] == 0
    assert led["accepted"] is True
    assert led["valid_lines"] == [
        "The model was trained on data.", "Another good sentence follows right here.",
        "A third fine line survives.",
    ]
    assert led["num_sentences"] == 3


def check_citation_removal(solve):
    # 引用先剥再查终止符与词数
    text = "\n".join([
        "Python [1] is widely used [12] today.",
        "See the section [edit] for details.",
        "This claim [citation needed] needs more work.",
    ])
    led = _compare(solve, text)
    assert led["valid_lines"] == [
        "Python  is widely used  today.",
        "See the section  for details.",
        "This claim  needs more work.",
    ]


def check_page_rejects(solve):
    base = ["The first line ends well.", "The second line ends well.",
            "The third line ends well."]
    # lorem ipsum：命中即整页拒
    t = "\n".join(base[:2] + ["Filler lorem ipsum text here."])
    led = _compare(solve, t)
    assert led["accepted"] is False and led["reject_reason"] == "filtered_page_loremipsum"
    assert led["valid_lines"] == []
    assert led["drop_counters"]["lines_no_endmark"] == 0  # 前两行都合法，未丢
    # 花括号：整页拒
    t = "\n".join(base[:2] + ["if (a) { return true; }."])
    led = _compare(solve, t)
    assert led["reject_reason"] == "filtered_page_squigglybracket"
    # badword：整页拒
    t = "\n".join(base[:2] + ["This page mentions the xxx topic."])
    led = _compare(solve, t, badwords=["xxx"])
    assert led["reject_reason"] == "filtered_page_badword"
    # 句数不足：2 < 3
    led = _compare(solve, "\n".join(base[:2]))
    assert led["reject_reason"] == "filtered_page_toofewsentences"


def check_badword_anchor_quirk(solve):
    # 上游正则 [^a-z](...)[^a-z]：词首/词尾/词中嵌套都不命中
    base = ["The first line ends well.", "The second line ends well.",
            "The third line ends well."]
    # "sex" 嵌在 essex 内不命中 → 页面通过
    t = "\n".join(base[:3] + ["The essex county records end here."])
    led = _compare(solve, t, badwords=["sex"])
    assert led["accepted"] is True
    # 前后都有非字母才命中
    t = "\n".join(base[:3] + ["Mentioning the sex topic openly."])
    led = _compare(solve, t, badwords=["sex"])
    assert led["reject_reason"] == "filtered_page_badword"


def check_word_length_boundary(solve):
    # 恰好 1000 字符的词保留，1001 丢行
    w1000 = "y" * 1000
    text = "\n".join([
        "Here is a %s token." % w1000,
        "Second line keeps going on.",
        "Third line ends right here.",
    ])
    led = _compare(solve, text)
    assert led["accepted"] is True
    w1001 = "z" * 1001
    text2 = "\n".join([
        "Here is a %s token." % w1001,
        "Second line keeps going on.",
        "Third line ends right here.",
    ])
    led2 = _compare(solve, text2)
    assert led2["drop_counters"]["lines_with_too_long_word"] == 1
    assert led2["num_sentences"] == 2


def check_min_words_boundary(solve):
    # 恰好 5 词保留，4 词丢
    five = "one two three four five."
    four = "one two three four."
    text = "\n".join([five, four, "Another fine line goes here.",
                      "And yet another line here."])
    led = _compare(solve, text)
    assert led["drop_counters"]["lines_too_short"] == 1
    assert led["num_sentences"] == 3


def check_dedup(solve):
    line1 = "The quick brown fox jumps over."
    line2 = "A different sentence lives here."
    line3 = "Yet another unique line exists."
    text = "\n".join([line1, line1, line2, line3])
    led = _compare(solve, text)
    assert led["drop_counters"]["lines_dedup"] == 1
    assert led["valid_lines"] == [line1, line2, line3]
    h1 = hashlib.md5(line1.strip().lower().encode("utf-8")).hexdigest()
    assert led["new_hashes"][0] == h1
    # 跨页去重：第二次调用带 seen_hashes，line1 直接被丢
    led2 = _compare(solve, "\n".join([
        line1, line2, line3,
        "Brand new content right here.",
        "Fresh material arrives here today.",
        "More unique text shows up.",
    ]), None, led["new_hashes"])
    assert led2["drop_counters"]["lines_dedup"] == 3  # 前三行全部命中跨页状态
    assert led2["valid_lines"][0] == "Brand new content right here."


def check_case_and_strip_dedup(solve):
    # 去重键 = strip + lower：大小写与首尾空白变体同键
    text = "\n".join([
        "The quick brown fox jumps over.",
        "  the QUICK brown fox jumps over.  ",
        "Another fine line goes here.",
        "One more good sentence too.",
    ])
    led = _compare(solve, text)
    assert led["drop_counters"]["lines_dedup"] == 1
    # 保留的是 strip 后的行（保留首个原行 strip 结果）
    assert led["valid_lines"][0] == "The quick brown fox jumps over."


def check_num_sentences_proxy(solve):
    # 题面钉死：每有效行 1 句；一行里多个终止符也只记 1
    text = "\n".join([
        "First line. With two marks!",
        "Second line of text here.",
        "Third line ends right here.",
    ])
    led = _compare(solve, text)
    assert led["num_sentences"] == 3


def check_invalid_inputs(solve):
    bad = [
        lambda: solve(123),
        lambda: solve("text.", badwords=["ok", ""]),
        lambda: solve("text.", seen_hashes="abc"),
        lambda: solve("text.", min_words_per_line=0),
        lambda: solve("text.", min_num_sentences=-1),
        lambda: solve("text.", max_word_length=True),
        lambda: solve("text.", min_num_sentences=2.5),
    ]
    for fn in bad:
        try:
            fn()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")
