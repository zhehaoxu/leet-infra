import re
import string


# ---------------------------------------------------------------- oracle
# 独立路径：切分用 finditer 手动切片（不调 re.split），关键词用单条交替正则，
# 去重用「成对排序 + 切片」而非逐条比较。

_SPLIT = re.compile(r"\n\d+\s?\. ")
_WS = re.compile(r"\s+")
_KW = re.compile(r"\b(image|images|graph|graphs|picture|pictures|file|files"
                 r"|map|maps|draw|plot|go to)\b", re.IGNORECASE)


def _cut(text):
    """按 '\\n<digits>[一个空白].<空格>' 切段，返回 (段列表)。"""
    marks = [(m.start(), m.end()) for m in _SPLIT.finditer(text)]
    if not marks:
        return [text]
    out = [text[:marks[0][0]]]
    for k, (_, end) in enumerate(marks):
        stop = marks[k + 1][0] if k + 1 < len(marks) else len(text)
        out.append(text[end:stop])
    return out


def _post(seg):
    text = _WS.sub(" ", seg).strip()
    text = text.strip()
    if text == "":
        return ""
    return text[:1].upper() + text[1:].lower()


def _why(inst):
    if inst == "":
        return "empty"
    n = len(inst.split())
    if n <= 3:
        return "too_short"
    if n > 150:
        return "too_long"
    if _KW.search(inst):
        return "keyword"
    if inst[:15] == "Write a program":
        return "write_a_program"
    if inst[0] in string.punctuation:
        return "punct"
    if not inst[0].isascii():
        return "non_ascii"
    return None


def _enc(instructions, classification):
    head = ("Come up with a series of classification tasks. "
            "Try to specify the possible output labels when possible.\n"
            if classification else "Come up with a series of tasks:\n")
    body = ""
    for i, s in enumerate(instructions):
        body += "%d. %s\n" % (i + 1, _WS.sub(" ", s).strip().rstrip(":"))
    return head + body + "%d." % (len(instructions) + 1)


def _py(response, seed_instructions, sims, finish_reason="stop",
        threshold=0.7, top_k=10, prompt_instructions=None,
        classification=False):
    segments = []
    if response is not None and finish_reason != "length":
        for seg in _cut(response):
            inst = _post(seg)
            why = _why(inst)
            segments.append({"raw": seg, "instruction": inst,
                             "kept": why is None, "reason": why})
    cands = [r["instruction"] for r in segments if r["kept"]]
    if sims is not None and len(sims) != len(cands):
        raise ValueError("sims 条数不符")
    if sims is None:
        return {"segments": segments, "candidates": cands,
                "dedup": [{"instruction": c, "accepted": True,
                           "max_similarity": None, "avg_similarity": 0.0,
                           "most_similar": {}} for c in cands],
                "instructions": list(cands), "prompt": None
                if prompt_instructions is None else _enc(prompt_instructions,
                                                         classification)}
    pool = list(seed_instructions)
    dedup, kept = [], []
    for inst, vec in zip(cands, sims):
        if len(vec) != len(pool):
            raise ValueError("池大小不符")
        accept = (not vec) or max(vec) <= threshold
        pairs = [(vec[i], i) for i in range(len(vec))]
        pairs.sort()
        top = pairs[-top_k:][::-1]
        most = {}
        if accept:
            for score, i in top:
                most[pool[i]] = score
            pool.append(inst)
            kept.append(inst)
        dedup.append({"instruction": inst, "accepted": accept,
                      "max_similarity": max(vec) if vec else None,
                      "avg_similarity": (sum(vec) / len(vec)) if vec else 0.0,
                      "most_similar": most})
    prompt = None
    if prompt_instructions is not None:
        prompt = _enc(prompt_instructions, classification)
    return {"segments": segments, "candidates": cands, "dedup": dedup,
            "instructions": kept, "prompt": prompt}


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


def _compare(solve, response, seed, sims, **kw):
    led = solve(response, seed, None if sims is None else [list(v) for v in sims], **kw)
    ref = _py(response, seed, sims, **kw)
    assert _eq(led, ref), f"\nled={led}\nref={ref}"
    return led


RESP = (
    "Here are some tasks:\n"
    "1. Write a short poem about the sea and the moon\n"      # 通过
    "2.   Tell me what you think about modern art today  \n"  # 通过（空白压缩）
    "3. Summarize the plot of a famous novel\n"               # keyword: plot
    "4. be nice\n"                                            # too_short（3 词）
    "5. Write a program to sort a list of numbers quickly\n"   # write_a_program
    "6. #hash tag everything in this sentence please now\n"    # punct
    "7. 描述一下如何用 Python 读取一个 CSV 文件并统计行数\n"     # non_ascii
)


def check_pipeline(solve):
    led = _compare(solve, RESP, ["seed one"], None)
    r = led["segments"]
    # 首段是文首没有 "\n<digits>. " 前导的那句："Here are some tasks:" 恰 4 词 -> 通过
    assert r[0]["reason"] is None, r[0]
    reasons = [x["reason"] for x in r]
    assert reasons[1] is None and reasons[2] is None
    assert reasons[3:] == ["keyword", "too_short", "write_a_program",
                          "punct", "non_ascii"], reasons
    assert led["candidates"][1] == "Write a short poem about the sea and the moon"
    assert led["candidates"][2] == "Tell me what you think about modern art today"
    assert all(x["reason"] is None for x in r if x["kept"])


def check_split_markers(solve):
    """切分正则 `\\n\\d+\\s?\\. ` 只要「换行 + 数字 + 至多一个空白 + 点 + 空格」。"""
    base = "preamble words here now\n"
    hits = {
        base + "9. Alpha beta gamma delta\n": 2,
        base + "9.  Alpha beta gamma delta\n": 2,   # 点后两个空格仍切
        base + "9 . Alpha beta gamma delta\n": 2,   # 数字与点之间一个空格
        base + "9\t. Alpha beta gamma delta\n": 2,  # 制表符算一个空白
        base + "9.Alpha beta gamma delta\n": 1,     # 点后必须跟空格
        base + "9 .Alpha beta gamma delta\n": 1,
        base + "9  . Alpha beta gamma delta\n": 1,  # 两个空白：\s? 只认一个
        base + "9\n. Alpha beta gamma delta\n": 2,   # 换行也算一个空白
        base + "x9. Alpha beta gamma delta\n": 1,   # 不是换行后的数字
    }
    for text, n in hits.items():
        led = _compare(solve, text, [], None)
        assert len(led["segments"]) == n, (text, len(led["segments"]))


def check_capitalize_quirk(solve):
    """官方 `capitalize()` 会把除首字母外全部小写，且它先于 "Write a program" 判定。"""
    text = ("p\n"
            "1. WRITE A PYTHON SCRIPT TO COUNT PRIMES\n"
            "2. Write A Program To Print Hello\n"
            "3. keep THIS case AS IS please\n")
    led = _compare(solve, text, [], None)
    assert led["segments"][1]["instruction"] == "Write a python script to count primes"
    assert led["segments"][1]["reason"] is None
    # capitalize 后正好命中 "Write a program" 前缀
    assert led["segments"][2]["reason"] == "write_a_program"
    assert led["segments"][3]["instruction"] == "Keep this case as is please"


def check_word_count(solve):
    def build(n):
        return "p\n1. " + " ".join(["w%d" % i for i in range(n)]) + "\n"
    led = _compare(solve, build(3), [], None)
    assert led["segments"][1]["reason"] == "too_short"
    led = _compare(solve, build(4), [], None)
    assert led["segments"][1]["reason"] is None
    led = _compare(solve, build(150), [], None)
    assert led["segments"][1]["reason"] is None
    led = _compare(solve, build(151), [], None)
    assert led["segments"][1]["reason"] == "too_long"


def check_keywords(solve):
    """`\\b(word)\\b` 大小写不敏感；词内不命中；"go to" 的空白由压缩步骤先归并。"""
    cases = {
        "List all IMAGES in the folder please": "keyword",
        "Describe the profile of a file系统": None,  # CJK 紧贴 "file" 打断 \b
        "Explain how to profile a python function": None,
        "Draw a cat sitting on a mat": "keyword",
        "Drawing a cat is a nice hobby": None,
        "Go  To the market and buy apples": "keyword",
        "GOTO the market and buy apples": None,
        "Plotting the curve of a function": None,
    }
    for text, want in cases.items():
        led = _compare(solve, "p\n1. " + text + "\n", [], None)
        assert led["segments"][1]["reason"] == want, (text, led["segments"][1])


def check_punct_and_ascii(solve):
    cases = {
        "42 is the answer to everything": None,
        "\\backslash the whole thing now": "punct",
        "«guillemets» around a phrase here": "non_ascii",
        "Описание задачи на русском языке": "non_ascii",
    }
    for text, want in cases.items():
        led = _compare(solve, "p\n1. " + text + "\n", [], None)
        assert led["segments"][1]["reason"] == want, (text, led["segments"][1])


def check_finish_length(solve):
    led = _compare(solve, RESP, ["s"], None, finish_reason="length")
    assert led["segments"] == [] and led["candidates"] == []
    led = _compare(solve, None, ["s"], None)
    assert led["segments"] == [] and led["instructions"] == []


def check_dedupe_threshold(solve):
    """阈值是严格 `>`；被拒的候选不进入池子，后续向量长度不增长。"""
    text = ("p\n"
            "1. Alpha beta gamma delta\n"
            "2. Bravo beta gamma delta\n"
            "3. Charlie beta gamma delta\n")
    led = _compare(solve, text, ["s1", "s2"],
                   [[0.5, 0.2], [0.7, 0.2, 0.1], [0.2, 0.1, 0.3, 0.3]])
    d = led["dedup"]
    assert [x["accepted"] for x in d] == [True, True, True]
    assert d[1]["max_similarity"] == 0.7  # 恰等于阈值不算超

    led = _compare(solve, text, ["s1", "s2"],
                   [[0.8, 0.2], [0.75, 0.2], [0.2, 0.9]])
    assert [x["accepted"] for x in led["dedup"]] == [False, False, False]
    assert led["instructions"] == []

    # 第 1 条被拒 -> 池子仍是 1，第 2、3 条的向量长度都只能是 1
    led = _compare(solve, text, ["s1"], [[0.9], [0.7], [0.3, 0.3]])
    assert [x["accepted"] for x in led["dedup"]] == [False, True, True]
    assert len(led["instructions"]) == 2


def check_pool_growth(solve):
    text = "p\n1. Alpha beta gamma delta\n2. Bravo beta gamma delta\n"
    # 第 1 条接受后池子变 2，第 2 条的向量必须 3 长（种子 2 + 已接受 1）
    led = _compare(solve, text, ["s1", "s2"], [[0.1, 0.1], [0.1, 0.1, 0.1]])
    assert len(led["instructions"]) == 2
    try:
        solve(text, ["s1", "s2"], [[0.1, 0.1], [0.1, 0.1]])
    except ValueError:
        pass
    else:
        raise AssertionError("池子增长后向量长度不足应抛 ValueError")


def check_most_similar(solve):
    """取末尾 top_k 再反转：分数降序，同分时下标大的在前。"""
    text = "p\n1. Alpha beta gamma delta\n"
    seed = ["s0", "s1", "s2", "s3"]
    led = _compare(solve, text, seed, [[0.1, 0.5, 0.5, 0.7]], top_k=3)
    most = led["dedup"][0]["most_similar"]
    assert list(most) == ["s3", "s2", "s1"], most      # 0.5 并列时下标大的在前
    assert most["s3"] == 0.7

    led = _compare(solve, text, seed, [[0.1, 0.5, 0.5, 0.7]], top_k=10)
    assert list(led["dedup"][0]["most_similar"]) == ["s3", "s2", "s1", "s0"]

    # 被拒的候选不给 most_similar
    led = _compare(solve, text, seed, [[0.1, 0.5, 0.5, 0.9]], top_k=3)
    assert led["dedup"][0]["most_similar"] == {}


def check_avg(solve):
    text = "p\n1. Alpha beta gamma delta\n2. Bravo beta gamma delta\n"
    led = _compare(solve, text, ["s1"], [[0.2], [0.1, 0.3]])
    assert abs(led["dedup"][0]["avg_similarity"] - 0.2) < 1e-12
    assert abs(led["dedup"][1]["avg_similarity"] - 0.2) < 1e-12
    # 空池子：max 无定义 -> 接受，avg 记 0.0
    led = _compare(solve, text, [], [[], [0.0]])
    assert [x["accepted"] for x in led["dedup"]] == [True, True]
    assert led["dedup"][0]["avg_similarity"] == 0.0
    assert led["dedup"][0]["max_similarity"] is None


def check_prompt(solve):
    led = _compare(solve, "", [], None, prompt_instructions=[
        "Summarize   this   text", "Rate it:", "Weird: :"])
    assert led["prompt"] == (
        "Come up with a series of tasks:\n"
        "1. Summarize this text\n"
        "2. Rate it\n"
        "3. Weird: \n"
        "4."), repr(led["prompt"])

    led = _compare(solve, "", [], None, prompt_instructions=["Do it:"],
                   classification=True)
    assert led["prompt"].startswith("Come up with a series of classification tasks.")
    assert led["prompt"].endswith("\n1. Do it\n2.")

    led = _compare(solve, "", [], None, prompt_instructions=[])
    assert led["prompt"] == "Come up with a series of tasks:\n1."

    led = _compare(solve, "", [], None)
    assert led["prompt"] is None


def check_rejections(solve):
    bad = [
        lambda: solve(5, [], []),
        lambda: solve("", [1], []),
        lambda: solve("", [], [], threshold=True),
        lambda: solve("", [], [], threshold=1.5),
        lambda: solve("", [], [], threshold=-0.1),
        lambda: solve("", [], [], threshold=float("nan")),
        lambda: solve("", [], [], top_k=0),
        lambda: solve("", [], [], top_k=True),
        lambda: solve("", [], [], top_k=2.0),
        lambda: solve("", [], "not a list"),
        lambda: solve("", [], [[]]),
        lambda: solve("p\n1. Alpha beta gamma delta\n", [], [[float("inf")]]),
        lambda: solve("", [], [], prompt_instructions="abc"),
        lambda: solve("", [], [], prompt_instructions=[1]),
        lambda: solve("", [], [], classification=1),
        lambda: solve("", [], [], finish_reason=1),
        lambda: solve("p\n1. Alpha beta gamma delta\n", [], [[0.1, 0.2]]),
    ]
    for fn in bad:
        try:
            fn()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")


def check_contract(solve):
    led = solve("", [], [])
    assert set(led) == {"segments", "candidates", "dedup", "instructions",
                        "prompt"}
    assert set(led["segments"][0]) == {"raw", "instruction", "kept", "reason"}
    seed = ["s1"]
    solve("p\n1. Alpha beta gamma delta\n", seed, [[0.1]])
    assert seed == ["s1"]  # 种子池不被就地改写


check_pipeline.example = True
check_prompt.example = True
