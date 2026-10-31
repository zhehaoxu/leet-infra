import random


def _oracle(
    text,
    min_doc_words=50,
    max_doc_words=100000,
    min_avg_word_length=3,
    max_avg_word_length=10,
    max_symbol_word_ratio=0.1,
    max_bullet_lines_ratio=0.9,
    max_ellipsis_lines_ratio=0.3,
    max_non_alpha_words_ratio=0.8,
    min_stop_words=2,
):
    """独立手写的 Gopher 账（datatrove GopherQualityFilter 口径，空白切词）。"""
    # 标点集是题面钉死的契约（与 starter 相同的 63 个字符），其余逻辑独立实现
    PUNCT = frozenset(
        "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
        "…—–·、，。！？；：""''《》【】（）〈〉「」『』"
    )
    STOP = {"the", "be", "to", "of", "and", "that", "have", "with"}
    words = text.split()
    nsw = [w for w in words if any(c not in PUNCT for c in w)]
    n = len(words)
    lines = text.splitlines()

    def bullets():
        c = 0
        for s in lines:
            t = s.lstrip()
            if t[:1] in ("•", "-"):
                c += 1
        return c / len(lines) if lines else 0.0

    def end_ell():
        c = 0
        for s in lines:
            t = s.rstrip()
            if t.endswith("...") or t.endswith("…"):
                c += 1
        return c / len(lines) if lines else 0.0

    avg = sum(len(w) for w in nsw) / len(nsw) if nsw else 0.0
    alpha = sum(1 for w in words if any(c.isalpha() for c in w)) / n if n else 0.0
    n_stop = len(STOP.intersection(words))

    def step(name, ok):
        if not ok:
            return name
        return None

    failed = None
    for name, ok in (
        ("gopher_short_doc", len(nsw) >= min_doc_words),
        ("gopher_long_doc", len(nsw) <= max_doc_words),
        ("gopher_avg_word_length", (min_avg_word_length <= avg <= max_avg_word_length) if nsw else True),
        ("gopher_too_many_hashes", (text.count("#") / n <= max_symbol_word_ratio) if n else True),
        (
            "gopher_too_many_ellipsis",
            ((text.count("...") + text.count("…")) / n <= max_symbol_word_ratio) if n else True,
        ),
        ("gopher_too_many_bullets", bullets() <= max_bullet_lines_ratio),
        ("gopher_too_many_end_ellipsis", end_ell() <= max_ellipsis_lines_ratio),
        ("gopher_below_alpha_threshold", alpha >= max_non_alpha_words_ratio),
        ("gopher_enough_stop_words", n_stop >= min_stop_words),
    ):
        failed = step(name, ok)
        if failed:
            break

    return {
        "passed": failed is None,
        "failed_rule": failed,
        "n_words": n,
        "n_non_symbol_words": len(nsw),
        "avg_word_length": avg,
        "bullet_lines_ratio": bullets(),
        "end_ellipsis_lines_ratio": end_ell(),
        "alpha_words_ratio": alpha,
        "n_stop_words": n_stop,
    }


def _compare(solve, text, **kw):
    got = solve(text, **kw)
    exp = _oracle(text, **kw)
    assert got["passed"] == exp["passed"], (text[:60], got["passed"], exp["passed"])
    assert got["failed_rule"] == exp["failed_rule"], (text[:60], got, exp)
    for k in (
        "n_words",
        "n_non_symbol_words",
        "bullet_lines_ratio",
        "end_ellipsis_lines_ratio",
        "alpha_words_ratio",
        "n_stop_words",
    ):
        assert got[k] == exp[k], (k, got[k], exp[k])
    assert abs(got["avg_word_length"] - exp["avg_word_length"]) < 1e-9
    return got


def check_anchor_pass(solve):
    # 80 词的正常英文文档，含 2 个停用词，全部规则通过
    body = (
        "The committee decided that the proposal should be reviewed by an independent "
        "board of experts before any funds are released to the research teams working "
        "on large scale training data pipelines across several countries and languages "
        "in order to reduce risk and improve the overall quality of the released models "
        "that people use with confidence today."
    )
    led = _compare(solve, body)
    assert led["passed"] is True and led["failed_rule"] is None
    assert led["n_words"] >= 50
    assert led["n_stop_words"] >= 2


def check_anchor_each_rule(solve):
    good = (
        "the model that we trained was evaluated on many benchmarks of code and text "
        "with care taken to avoid contamination between splits and to report honest "
        "numbers for every single task and setting that the team measured during the "
        "study and the results were shared with reviewers to have an open discussion "
        "of the findings to the public and of the limitations and of the failures"
    )
    assert _compare(solve, good)["passed"] is True

    # short_doc：13 个非标点词（"#a#b" 是纯标点词）
    led = _compare(solve, "the model that we have is small and it works fine today")
    assert led["failed_rule"] == "gopher_short_doc"

    # 顺序优先级：短文档 + hash 超标 → 先报 short_doc
    led = _compare(solve, "### the model that we have is small and it works fine today")
    assert led["failed_rule"] == "gopher_short_doc"
    assert led["checks"]["symbol_word_ratio"] is False

    # long_doc：把上限调小
    led = _compare(solve, good, max_doc_words=10)
    assert led["failed_rule"] == "gopher_long_doc"

    # avg_word_length：全是一两个字母的词（文档长度过关：50+ 词）
    tiny = " ".join(["ab"] * 60)
    led = _compare(solve, tiny)
    assert led["failed_rule"] == "gopher_avg_word_length"
    # 平均词长超上限
    longw = " ".join(["abcdefghijkl"] * 60)
    led = _compare(solve, longw)
    assert led["failed_rule"] == "gopher_avg_word_length"

    # hashes：'#' 计数 / 词数 > 0.1（文档长度先过关）
    text = " ".join(["the"] * 60 + ["#", "#", "#", "#", "#", "#", "#", "#"])
    led = _compare(solve, text)
    assert led["failed_rule"] == "gopher_too_many_hashes"

    # ellipsis：'…' 与 '...' 都计入分子
    text = " ".join(["the"] * 60 + ["..."] * 4 + ["…"] * 3)
    led = _compare(solve, text)
    assert led["failed_rule"] == "gopher_too_many_ellipsis"

    # bullets：> 90% 行以 • 或 - 开头（总词数先过关）
    text = "\n".join(
        ["- item %d of the list and that have with to be" % i for i in range(5)]
    )
    led = _compare(solve, text)
    assert led["failed_rule"] == "gopher_too_many_bullets"
    assert abs(led["bullet_lines_ratio"] - 1.0) < 1e-12

    # end_ellipsis：> 30% 行以 ... 或 … 结尾
    text = "\n".join(
        ["the to of and that with have be list token here still %d..." % i for i in range(3)]
        + ["a final plain line of words here to close the document for good today"]
    )
    led = _compare(solve, text)
    assert led["failed_rule"] == "gopher_too_many_end_ellipsis"
    assert abs(led["end_ellipsis_lines_ratio"] - 0.75) < 1e-12

    # alpha：80% 词无字母（4 字母数字词不触发 avg 规则）
    text = " ".join(["1234", "5678", "9012", "!!"] * 16 + ["the", "of"])
    led = _compare(solve, text)
    assert led["failed_rule"] == "gopher_below_alpha_threshold"

    # stop words：交集 < 2
    text = " ".join(["alpha"] * 60)
    led = _compare(solve, text)
    assert led["failed_rule"] == "gopher_enough_stop_words"
    assert led["n_stop_words"] == 0


def check_boundaries_pass(solve):
    # 边界等于阈值时一律通过（上游用严格不等号判拒）
    body = (
        "the data that we have collected from the web is noisy and requires a lot of "
        "careful cleaning and the tools that we built for this purpose are fast and "
        "simple and they scale to trillions of tokens with modest compute and the "
        "quality of the final corpus depends on every single rule and threshold that "
        "the pipeline applies to each document in turn to keep only the good parts of"
    )
    assert len(body.split()) == 71
    # 词数 71 > 70 阈值 → long_doc 拒；用 max_doc_words=71 恰好通过
    led = _compare(solve, body, max_doc_words=71)
    assert led["passed"] is True
    # 恰好 60 词：max_doc_words=60 → 60 <= 60 通过
    words = body.split()[:60]
    led = _compare(solve, " ".join(words), max_doc_words=60)
    assert led["passed"] is True
    # avg 恰好等于上限：全 10 字母词 + 2 停用词
    ten = " ".join(["helloworld"] * 49 + ["the", "of"])
    led = _compare(solve, ten, max_avg_word_length=10)
    assert led["passed"] is True
    # hash ratio 恰好等于 0.1：60 词中 6 个 #（非标点词 54 也须 >= 50）
    led = _compare(solve, " ".join(["data"] * 51 + ["the", "the", "of"] + ["#"] * 6))
    assert led["passed"] is True
    # 略超：60 词中 9 个 # → 0.15 > 0.1 拒
    led = _compare(solve, " ".join(["data"] * 51 + ["the", "the", "of"] + ["#"] * 9))
    assert led["failed_rule"] == "gopher_too_many_hashes"


def check_punctuation_words_excluded(solve):
    # 纯标点词不计入 doc words，但计入分母 n_words（symbol ratio 的分母是全部词）
    text = " ".join(["the", "of"] * 30 + ["???"])
    led = _compare(solve, text, min_doc_words=60)
    assert led["n_words"] == 61
    assert led["n_non_symbol_words"] == 60
    # avg word length 只统计非标点词
    text = " ".join(["the"] * 29 + ["that"] * 29 + ["??????"] * 2)
    led = _compare(solve, text)
    assert led["passed"] is True  # 纯标点词不进 avg，avg=3.5 在界内，stop 种类 the/that=2；?????? 也不匹配 end ellipsis
    led2 = _compare(solve, text, max_avg_word_length=3)
    assert led2["failed_rule"] == "gopher_avg_word_length"


def check_bullet_and_ellipsis_lines(solve):
    # lstrip/rstrip 语义：前导空白后是 bullet 仍算；1/4 行是 bullet（<=0.9 通过）
    text = "\n".join(
        [
            "   - the of and that with have be to list item one two four five six",
            "plain narrative line of body text describing a model training run of tokens now",
            "another plain narrative line of body text describing evaluation of quality and depth",
            "yet another plain narrative line of body text describing data of coverage and scope",
        ]
    )
    led = _compare(solve, text)
    assert abs(led["bullet_lines_ratio"] - 0.25) < 1e-12
    assert led["passed"] is True
    # 行内 - 不算 bullet；行尾 … 算 end ellipsis（1/5 <= 0.3 通过）
    text2 = "\n".join(
        [
            "a - b of and that with have be to list item one",
            "x token here and there in the mix …",
            "plain line of the body text for this document",
            "more plain body text lines to fill the word budget here",
            "and even more plain body text lines to pass every rule",
        ]
    )
    led2 = _compare(solve, text2)
    assert abs(led2["bullet_lines_ratio"] - 0.0) < 1e-12
    assert abs(led2["end_ellipsis_lines_ratio"] - 0.2) < 1e-12
    assert led2["passed"] is True
    # 行尾空白后的 ... 仍算 end ellipsis（rstrip 语义）
    text3 = "\n".join(
        [
            "the to of and that with have be to list item one two ...   ",
            "plain line of the body text for this document",
            "more plain body text lines to fill the word budget here",
            "and even more plain body text lines to pass every rule",
            "final plain body text line closing the whole document out",
        ]
    )
    led3 = _compare(solve, text3)
    assert abs(led3["end_ellipsis_lines_ratio"] - 0.2) < 1e-12
    assert led3["passed"] is True


def check_custom_params(solve):
    body = (
        "the data that we have collected from the web is noisy and requires a lot of "
        "careful cleaning and the tools that we built for this purpose are fast and "
        "simple and they scale to trillions of tokens with modest compute and the "
        "quality of the final corpus depends on every single rule and threshold that "
        "the pipeline applies to each document in turn to keep only the good parts of"
    )
    # 放宽 min_doc_words 后原本的短文档过了长度关，撞上停用词规则（交集 4 < 5）
    led = _compare(solve, "the model that we have is small and it works fine today",
                   min_doc_words=10, min_stop_words=5)
    assert led["failed_rule"] == "gopher_enough_stop_words"
    # min_stop_words 提到 8：长文的停用词种类只有 7 种 → 拒
    led2 = _compare(solve, body, min_stop_words=8)
    assert led2["failed_rule"] == "gopher_enough_stop_words"
    # 停用词用自定义集不可能（STOP_WORDS 钉死），但 min_stop_words=0 恒过该条
    led3 = _compare(solve, " ".join(["xyz"] * 55), min_stop_words=0)
    assert led3["checks"]["stop_words"] is True


def check_empty_text(solve):
    led = _compare(solve, "")
    assert led["passed"] is False
    assert led["failed_rule"] == "gopher_short_doc"
    assert led["n_words"] == 0
    assert led["n_stop_words"] == 0


def check_random_against_oracle(solve):
    rng = random.Random(656)
    vocab = (
        ["the", "of", "and", "to", "that", "with", "have", "be", "model", "data",
         "training", "pipeline", "quality", "filter", "rules", "text", "token",
         "ab", "xy", "abcde", "123", "4567", "!!!!!", "...", "…", "#", "###",
         "-", "•", "——"]
    )
    for _ in range(50):
        n = rng.randint(0, 90)
        words = [rng.choice(vocab) for _ in range(n)]
        if rng.random() < 0.5:
            lines = []
            i = 0
            while i < len(words):
                j = min(len(words), i + rng.randint(1, 12))
                lines.append(" ".join(words[i:j]))
                i = j
            text = "\n".join(lines)
        else:
            text = " ".join(words)
        kw = {}
        if rng.random() < 0.3:
            kw["min_doc_words"] = rng.choice([0, 1, 10, 30])
        if rng.random() < 0.3:
            kw["max_symbol_word_ratio"] = rng.choice([0.05, 0.2, 0.5])
        if rng.random() < 0.3:
            kw["max_bullet_lines_ratio"] = rng.choice([0.0, 0.5])
        if rng.random() < 0.3:
            kw["max_ellipsis_lines_ratio"] = rng.choice([0.0, 0.5])
        if rng.random() < 0.3:
            kw["min_stop_words"] = rng.choice([0, 1, 3])
        _compare(solve, text, **kw)
