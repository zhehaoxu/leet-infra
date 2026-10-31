from collections import Counter


def _py(text, thresholds=None):
    """独立重写的 oracle：行/段重复用双循环计数，n-gram 覆盖用逐字符
    boolean mask 标记（与 solution 的区间合并不同实现路径）。"""
    if not isinstance(text, str):
        raise ValueError("text must be a string")

    DEFAULT = {
        "duplicate_line_fraction": 0.30,
        "duplicate_paragraph_fraction": 0.30,
        "duplicate_line_character_fraction": 0.20,
        "duplicate_paragraph_character_fraction": 0.20,
        "top_2_gram_character_fraction": 0.20,
        "top_3_gram_character_fraction": 0.18,
        "top_4_gram_character_fraction": 0.16,
        "duplicate_5_gram_character_fraction": 0.15,
        "duplicate_6_gram_character_fraction": 0.14,
        "duplicate_7_gram_character_fraction": 0.13,
        "duplicate_8_gram_character_fraction": 0.12,
        "duplicate_9_gram_character_fraction": 0.11,
        "duplicate_10_gram_character_fraction": 0.10,
    }
    if thresholds is not None:
        for k, v in thresholds.items():
            if k not in DEFAULT:
                raise ValueError("unknown metric")
            if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0 or v > 1:
                raise ValueError("bad threshold")
    thr = dict(DEFAULT)
    if thresholds:
        thr.update(thresholds)

    lines = text.splitlines()
    # 行重复：双循环统计实例
    dup_line_n = 0
    dup_line_chars = 0
    for i, l in enumerate(lines):
        dup = False
        for j, m in enumerate(lines):
            if j != i and l == m:
                dup = True
                break
        if dup:
            dup_line_n += 1
            dup_line_chars += len(l)
    line_chars_total = 0
    for l in lines:
        line_chars_total += len(l)
    dup_line_frac = dup_line_n / len(lines) if lines else 0.0
    dup_line_char = dup_line_chars / line_chars_total if line_chars_total else 0.0

    # 段落：空行分隔
    paras = []
    block = []
    for l in lines:
        if l.strip() == "":
            if block:
                paras.append(block)
                block = []
        else:
            block.append(l)
    if block:
        paras.append(block)
    para_strs = ["\n".join(b) for b in paras]
    para_chars = [sum(len(x) for x in b) for b in paras]
    dup_para_n = 0
    dup_para_chars = 0
    for i, p in enumerate(para_strs):
        dup = False
        for j, q in enumerate(para_strs):
            if j != i and p == q:
                dup = True
                break
        if dup:
            dup_para_n += 1
            dup_para_chars += para_chars[i]
    para_chars_total = sum(para_chars)
    dup_para_frac = dup_para_n / len(para_strs) if para_strs else 0.0
    dup_para_char = dup_para_chars / para_chars_total if para_chars_total else 0.0

    # 词与 n-gram：覆盖用逐字符 mask
    words = []
    i, n = 0, len(text)
    while i < n:
        if text[i].isspace():
            i += 1
            continue
        j = i
        while j < n and not text[j].isspace():
            j += 1
        words.append((text[i:j], i, j))
        i = j

    def cover_fraction(spans):
        if not spans or len(text) == 0:
            return 0.0
        mask = [False] * len(text)
        for s, e in spans:
            for k in range(s, e):
                mask[k] = True
        return sum(1 for m in mask if m) / len(text)

    metrics = {
        "duplicate_line_fraction": dup_line_frac,
        "duplicate_paragraph_fraction": dup_para_frac,
        "duplicate_line_character_fraction": dup_line_char,
        "duplicate_paragraph_character_fraction": dup_para_char,
    }
    for n in (2, 3, 4):
        info = {}
        for i in range(len(words) - n + 1):
            key = tuple(words[i + t][0] for t in range(n))
            info.setdefault(key, {"count": 0, "first": words[i][1], "spans": []})
            info[key]["count"] += 1
            info[key]["spans"].append((words[i][1], words[i + n - 1][2]))
        best_key = None
        best = None
        for key, d in info.items():
            cand = (d["count"], -d["first"])
            if best is None or cand > best:
                best = cand
                best_key = key
        metrics[f"top_{n}_gram_character_fraction"] = (
            cover_fraction(info[best_key]["spans"]) if best_key else 0.0
        )
    for n in range(5, 11):
        spans = []
        for i in range(len(words) - n + 1):
            key = tuple(words[i + t][0] for t in range(n))
            cnt = 0
            for j in range(len(words) - n + 1):
                if tuple(words[j + t][0] for t in range(n)) == key:
                    cnt += 1
            if cnt >= 2:
                spans.append((words[i][1], words[i + n - 1][2]))
        metrics[f"duplicate_{n}_gram_character_fraction"] = cover_fraction(spans)

    order = list(DEFAULT.keys())
    reason = None
    for name in order:
        if metrics[name] > thr[name]:
            reason = name
            break
    return {"accepted": reason is None, "reject_reason": reason, "metrics": metrics}


def _approx(a, b):
    return abs(a - b) < 1e-9


def _compare(solve, text, thresholds=None):
    led = solve(text, thresholds)
    ref = _py(text, thresholds)
    assert led == ref, f"\nled={led}\nref={ref}"
    return led


def check_all_identical_lines(solve):
    # 三组相同行重复：行实例 fraction = 1.0，Table A1 顺序第一个超限
    text = "alpha beta\ndelta\ngamma\nalpha beta\ndelta\ngamma\nalpha beta\ndelta\ngamma"
    led = _compare(solve, text)
    assert led["accepted"] is False
    assert led["reject_reason"] == "duplicate_line_fraction"
    assert _approx(led["metrics"]["duplicate_line_fraction"], 1.0)
    assert _approx(led["metrics"]["duplicate_line_character_fraction"], 1.0)


def check_line_fraction_boundary(solve):
    # A 出现 2 次：2/10 = 0.2
    lines2 = ["A", "bb", "A", "dd", "ee", "ff", "gg", "hh", "ii", "jj"]
    led = _compare(solve, "\n".join(lines2))
    assert _approx(led["metrics"]["duplicate_line_fraction"], 0.2)
    # A 出现 3 次：3/10 = 0.30 恰好等于默认阈值（surpasses = 严格大于，不触发）
    lines3 = ["A", "bb", "A", "A", "dd", "ee", "ff", "gg", "hh", "ii"]
    text3 = "\n".join(lines3)
    led3 = _compare(solve, text3)
    assert _approx(led3["metrics"]["duplicate_line_fraction"], 0.3)
    assert led3["metrics"]["duplicate_line_fraction"] <= 0.30
    # 阈值调到 0.29 → 0.3 > 0.29 触发；调到 0.31 → 行账不再触发
    led4 = _compare(solve, text3, {"duplicate_line_fraction": 0.29})
    assert led4["reject_reason"] == "duplicate_line_fraction"
    led5 = _compare(solve, text3, {"duplicate_line_fraction": 0.31})
    assert led5["reject_reason"] != "duplicate_line_fraction"
    # 大小写与尾随空白是不同行（精确匹配）：不构成重复
    text_case = "\n".join(["A", "a", "A ", "bb", "cc", "dd", "ee", "ff", "gg", "hh"])
    led6 = _compare(solve, text_case)
    assert _approx(led6["metrics"]["duplicate_line_fraction"], 0.0)


def check_line_char_fraction(solve):
    # 行实例占比低（2/10）但重复行字符占比高 → 触发 duplicate_line_character_fraction
    long_dup = "x" * 20
    text = "\n".join([long_dup, "a", "b", "c", "d", "e", "f", "g", long_dup, "h"])
    led = _compare(solve, text)
    assert led["reject_reason"] == "duplicate_line_character_fraction"
    assert _approx(led["metrics"]["duplicate_line_fraction"], 0.2)
    # 重复行字符 40，全部行字符 48
    assert _approx(led["metrics"]["duplicate_line_character_fraction"], 40 / 48)


def check_paragraph_metrics(solve):
    # 注意 quirk：空行本身也是行且参与行重复（原样精确匹配）。
    # text2 有 5 个空行 + "x"×2 = 7 个重复行实例 / 15 行 → 行账先触发（Table A1 顺序）。
    text = "x\n\nx\n\na\nb\n\nc\nd\n\ne\nf\n\ng\nh"
    led = _compare(solve, text)
    assert led["reject_reason"] == "duplicate_line_fraction"
    assert _approx(led["metrics"]["duplicate_line_fraction"], 7 / 15)
    # 段账的值仍然正确：6 段中 "x" 段 ×2
    assert _approx(led["metrics"]["duplicate_paragraph_fraction"], 2 / 6)
    assert _approx(led["metrics"]["duplicate_paragraph_character_fraction"], 2 / 10)


def check_paragraph_reason(solve):
    # 行账全部压到阈值内、只有段账超限的构造：
    # 6 段（"x" ×2 + 4 段各 5 行唯一词行），27 行中重复行实例仅 7/25.9%
    s3 = ["aa bb", "cc dd", "ee ff", "gg hh", "ii jj"]
    s4 = ["kk ll", "mm nn", "oo pp", "qq rr", "ss tt"]
    s5 = ["uu vv", "ww xx", "yy zz", "ab cd", "ef gh"]
    s6 = ["ij kl", "mn op", "qr st", "uv wx", "yz ab"]
    text = "x\n\nx\n\n" + "\n\n".join("\n".join(s) for s in (s3, s4, s5, s6))
    led = _compare(solve, text)
    m = led["metrics"]
    assert _approx(m["duplicate_line_fraction"], 7 / 27)
    assert m["duplicate_line_fraction"] <= 0.30
    assert m["duplicate_line_character_fraction"] <= 0.20
    assert _approx(m["duplicate_paragraph_fraction"], 2 / 6)
    assert led["reject_reason"] == "duplicate_paragraph_fraction"


def check_para_char_dominated_by_line(solve):
    # 机制事实：段重复必导致其组成行重复，且分母同源，
    # 故 duplicate_line_character_fraction >= duplicate_paragraph_character_fraction，
    # 段字符账永远不会先于行字符账触发（行账在 Table A1 中序号更靠前）。
    long_line = "w" * 30
    text = (long_line + "\n" + "other line here\n" +
            long_line + "\n" + "another unique row\n" +
            "yet one more line\n" + "and the final line")
    led = _compare(solve, text)
    m = led["metrics"]
    assert m["duplicate_line_character_fraction"] >= m["duplicate_paragraph_character_fraction"]
    assert led["reject_reason"] in ("duplicate_line_fraction",
                                    "duplicate_line_character_fraction")


def check_top_ngram_first_occurrence_tie(solve):
    # 单行文档（无行/段重复）；所有 2-gram 均出现 1 次 → 全并列，
    # 取文档中首次出现位置最早者 (ab, cdefg)
    text = "ab cdefg xy q rstuv"
    led = _compare(solve, text)
    assert led["reject_reason"] == "top_2_gram_character_fraction"
    # (ab,cdefg) 覆盖 [0,8) = 8 字符；len(text)=19
    assert _approx(led["metrics"]["top_2_gram_character_fraction"], 8 / 19)
    # top-3gram 首现 (ab,cdefg,xy) 覆盖 [0,11) = 11/19
    assert _approx(led["metrics"]["top_3_gram_character_fraction"], 11 / 19)
    # top-4gram (ab,cdefg,xy,q) 覆盖 [0,13) = 13/19
    assert _approx(led["metrics"]["top_4_gram_character_fraction"], 13 / 19)


def check_duplicate_ngram_overlap(solve):
    # 5-gram "pp qq rr ss t" 在长文档中重复两次，其余部分是唯一填充词：
    # top2 = 10/150、top3 = 16/150、top4 = 22/150 均不超阈值，
    # dup5 = 26/150 > 0.15 → 触发 duplicate_5_gram_character_fraction
    filler1 = "a" * 61
    filler2 = "b" * 60
    text = "pp qq rr ss t " + filler1 + " pp qq rr ss t " + filler2
    assert len(text) == 150
    led = _compare(solve, text)
    assert led["reject_reason"] == "duplicate_5_gram_character_fraction"
    assert _approx(led["metrics"]["top_2_gram_character_fraction"], 10 / 150)
    assert _approx(led["metrics"]["top_3_gram_character_fraction"], 16 / 150)
    assert _approx(led["metrics"]["top_4_gram_character_fraction"], 22 / 150)
    # 两处出现 span (0,13) 与 (76,89)，不重叠 → 覆盖 26
    assert _approx(led["metrics"]["duplicate_5_gram_character_fraction"], 26 / 150)


def check_periodic_overlap_merge(solve):
    # 周期文档：5-gram 实例两两重叠，覆盖必须按区间合并（重叠不重复计）
    text = "a b c d e a b c d e a b c d e"
    led = _compare(solve, text)
    # 11 个 5-gram 实例，span (2s, 2s+10) s=0..10，末词 end=29 →
    # 合并覆盖 [0,29) = 全文档 → 29/29 = 1.0
    # （未合并的朴素累加是 11×10/29 ≈ 3.79，远大于 1.0）
    assert _approx(led["metrics"]["duplicate_5_gram_character_fraction"], 1.0)
    assert led["metrics"]["duplicate_5_gram_character_fraction"] <= 1.0
    # 周期文本里 (a,b) 出现 3 次 → top_2 先触发（Table A1 顺序账）
    assert led["reject_reason"] == "top_2_gram_character_fraction"
    assert _approx(led["metrics"]["top_2_gram_character_fraction"], 9 / 29)


def check_threshold_override(solve):
    # 覆盖行 fraction 阈值后落到下一个超限度量（行字符 fraction = 1.0）
    text = "alpha beta\ndelta\ngamma\nalpha beta\ndelta\ngamma\nalpha beta\ndelta\ngamma"
    led = _compare(solve, text, {"duplicate_line_fraction": 1.0})
    assert led["reject_reason"] == "duplicate_line_character_fraction"
    # 行 fraction 阈值放宽到 1.1 非法（>1）
    try:
        solve(text, {"duplicate_line_fraction": 1.1})
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def check_empty_and_short(solve):
    led = _compare(solve, "")
    assert led["accepted"] is True and led["reject_reason"] is None
    for v in led["metrics"].values():
        assert v == 0.0
    # 纯空白：无段无词
    led2 = _compare(solve, "   \n  ")
    assert led2["accepted"] is True
    # 词数不足 n：dup n-gram 度量记 0.0，top n-gram 仍取首现
    led3 = _compare(solve, "only one sentence here")
    assert led3["metrics"]["duplicate_5_gram_character_fraction"] == 0.0
    assert led3["metrics"]["top_4_gram_character_fraction"] > 0.0


def check_line_char_boundary(solve):
    # 恰等锚点：等长行 + "aa bb cc" 整行出现 2 次 →
    # 行 fraction = 2/10 = 0.2 ≤ 0.30、行字符 fraction = 16/80 = 0.2 恰等于阈值
    # （surpasses 是严格大于 → 不触发）；行重复使 (aa,bb)/(aa,bb,cc) count=2，
    # 但 top 覆盖被文档长度稀释到阈值内 → accepted
    rows = ["aa bb cc", "dd ee ff", "gg hh ii", "jj kk ll", "mm nn oo",
            "pp qq rr", "ss tt uu", "vv ww xx", "aa bb cc", "ab cd ef"]
    text = "\n".join(rows)
    led = _compare(solve, text)
    m = led["metrics"]
    assert _approx(m["duplicate_line_fraction"], 0.2)
    assert _approx(m["duplicate_line_character_fraction"], 0.2)
    # (aa,bb) count=2：spans (0,5) 与 (72,77) → 10/89
    assert _approx(m["top_2_gram_character_fraction"], 10 / 89)
    # (aa,bb,cc) count=2：spans (0,8) 与 (72,80) → 16/89 ≈ 0.1798 ≤ 0.18 恰不超
    assert _approx(m["top_3_gram_character_fraction"], 16 / 89)
    # 无重复 4-gram（两处 A 行后继词不同）→ 首现 (aa,bb,cc,dd) span (0,11)
    assert _approx(m["top_4_gram_character_fraction"], 11 / 89)
    assert led["accepted"] is True and led["reject_reason"] is None


def check_invalid_inputs(solve):
    bad = [
        lambda: solve(123),
        lambda: solve("x", thresholds=["duplicate_line_fraction"]),
        lambda: solve("x", thresholds={"nope": 0.1}),
        lambda: solve("x", thresholds={"duplicate_line_fraction": True}),
        lambda: solve("x", thresholds={"duplicate_line_fraction": "0.1"}),
        lambda: solve("x", thresholds={"top_2_gram_character_fraction": 1.5}),
        lambda: solve("x", thresholds={"duplicate_5_gram_character_fraction": -0.1}),
    ]
    for fn in bad:
        try:
            fn()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")
