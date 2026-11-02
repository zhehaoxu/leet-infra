"""Chain of Agents 串行接力账的检查点。

每个检查点都拿一份**独立实现**的 oracle（另一套分块/接力写法）与
`solve` 比对，再补若干硬断言钉住容易写错的边界。
"""


# ---------------------------------------------------------------- oracle --
def _bucket(text, k):
    """另一种写法：先按段落分组，超窗段落预切成整窗块。返回词列表。"""
    buckets, cur = [], []
    for para in text.split("\n\n"):
        if not para.strip():
            continue
        words = para.split()
        if len(cur) + len(words) <= k:
            cur = cur + words
            continue
        if cur:
            buckets.append(cur)
            cur = []
        while len(words) > k:
            buckets.append(words[:k])
            words = words[k:]
        cur = words
    if cur:
        buckets.append(cur)
    return buckets


def _oracle(source, query, worker_out, manager_out, k=8, max_cu=None,
            window=None, instr_words=0, manager_instr_words=0,
            order="left-to-right"):
    if k < 1:
        raise ValueError("k")
    if max_cu is not None and max_cu < 1:
        raise ValueError("max_cu")
    if order not in ("left-to-right", "right-to-left"):
        raise ValueError("order")

    buckets = _bucket(source, k)
    if not buckets:
        raise ValueError("empty")
    if len(worker_out) != len(buckets):
        raise ValueError("arity")

    texts = [" ".join(w) for w in buckets]
    cursor = 0
    chunk_rows = []
    for i, text in enumerate(texts):
        n = len(text.split())
        chunk_rows.append({"idx": i, "text": text, "words": n,
                           "start": cursor, "end": cursor + n})
        cursor += n

    walk = list(range(len(buckets)))
    if order == "right-to-left":
        walk = walk[::-1]

    q_n = len(query.split())
    cu_rows, cu = [], ""
    for pos in range(len(walk)):
        idx = walk[pos]
        chunk_n = len(texts[idx].split())
        in_n = instr_words + len(cu.split()) + chunk_n + q_n
        if window is not None and in_n > window:
            raise ValueError("overflow")
        raw = worker_out[idx]
        raw_n = len(raw.split())
        if raw_n < 1:
            nxt, cut, reused = cu, False, True
        elif max_cu is not None and raw_n > max_cu:
            nxt, cut, reused = " ".join(raw.split()[:max_cu]), True, False
        else:
            nxt, cut, reused = raw, False, False
        cu_rows.append({
            "chunk_idx": idx,
            "chunk_words": chunk_n,
            "cu_in_words": len(cu.split()),
            "raw_words": raw_n,
            "cu_out_words": len(nxt.split()),
            "input_words": in_n,
            "truncated": cut,
            "reused": reused,
        })
        cu = nxt

    m_n = manager_instr_words + len(cu.split()) + q_n
    if window is not None and m_n > window:
        raise ValueError("manager overflow")

    return {
        "num_chunks": len(buckets),
        "chunks": chunk_rows,
        "order": walk,
        "cus": _cus_of(cu_rows, worker_out, buckets, walk, max_cu),
        "workers": cu_rows,
        "cu_l": cu,
        "manager_cus": [cu],
        "manager_input_words": m_n,
        "answer": manager_out,
    }


def _cus_of(cu_rows, worker_out, buckets, walk, max_cu):
    """按处理顺序重放一遍，得到每轮 CU 文本。"""
    out, cu = [], ""
    for idx in walk:
        raw = worker_out[idx]
        raw_n = len(raw.split())
        if raw_n < 1:
            nxt = cu
        elif max_cu is not None and raw_n > max_cu:
            nxt = " ".join(raw.split()[:max_cu])
        else:
            nxt = raw
        out.append(nxt)
        cu = nxt
    return out


def _same(solve, *args, **kwargs):
    got = solve(*args, **kwargs)
    want = _oracle(*args, **kwargs)
    assert got == want, "账本不一致\n got=%r\n exp=%r" % (got, want)
    return got


# ------------------------------------------------------------- fixtures --
_LONG = "alpha beta gamma\n\ndelta epsilon\n\nzeta eta theta iota"
_W3 = ["CU one", "CU two here", "CU three four"]


# ------------------------------------------------------------ checks ----
def check_basic_relay(solve):
    """三块基本链：CU_0 为空、逐轮只看上一个 CU、manager 只看最后一个。"""
    r = _same(solve, _LONG, "what", _W3, "final", k=4)
    assert r["num_chunks"] == 3
    assert [c["text"] for c in r["chunks"]] == [
        "alpha beta gamma", "delta epsilon", "zeta eta theta iota"]
    assert [c["words"] for c in r["chunks"]] == [3, 2, 4]
    assert [(c["start"], c["end"]) for c in r["chunks"]] == [
        (0, 3), (3, 5), (5, 9)]
    assert r["cus"] == _W3
    assert r["order"] == [0, 1, 2]
    # 第一轮的 cu_in 必然是 0：CU_0 是空串
    assert r["workers"][0]["cu_in_words"] == 0
    assert [w["cu_in_words"] for w in r["workers"]] == [0, 2, 3]
    assert [w["input_words"] for w in r["workers"]] == [4, 5, 8]
    assert r["cu_l"] == "CU three four"
    assert r["manager_cus"] == ["CU three four"]
    assert r["answer"] == "final"


def check_split_paragraphs(solve):
    """段落合并、纯空白段跳过、段内换行与多余空格归一成单空格。"""
    r = _same(solve, "a  b\n\n \t \n\nc\nd\n\n e ", "q", ["x"], "z", k=10)
    assert [c["text"] for c in r["chunks"]] == ["a b c d e"]
    assert r["chunks"][0]["words"] == 5
    # 段落之间不留分隔符：`\n\n` 被整段吃掉
    assert "\n" not in r["chunks"][0]["text"]
    # 单段落 + 恰好装满窗口
    r = _same(solve, "a b c\n\nd", "q", ["x", "y"], "z", k=3)
    assert [c["text"] for c in r["chunks"]] == ["a b c", "d"]


def check_split_hard_cut(solve):
    """超窗段落硬切成整窗块；尾巴不产空块；块词数恒 <= k。"""
    r = _same(solve, "a b c d e f g", "q", ["1", "2", "3"], "z", k=3)
    assert [c["text"] for c in r["chunks"]] == ["a b c", "d e f", "g"]
    assert [(c["start"], c["end"]) for c in r["chunks"]] == [
        (0, 3), (3, 6), (6, 7)]
    # 恰好整窗：切完不剩尾巴 → 不产生空块
    r = _same(solve, "a b c d", "q", ["1", "2"], "z", k=2)
    assert [c["text"] for c in r["chunks"]] == ["a b", "c d"]
    assert r["num_chunks"] == 2
    # 满块滞留：攒满 k 词后不立刻落盘，要等下一段
    r = _same(solve, "a b\n\nc d\n\ne", "q", ["1", "2"], "z", k=3)
    assert [c["text"] for c in r["chunks"]] == ["a b", "c d e"]
    assert all(c["words"] <= 3 for c in r["chunks"])


def check_cu_reuse(solve):
    """空生成 = 本块无相关信息：沿用上一个 CU，不是清空。"""
    r = _same(solve, "a b\n\nc d\n\ne f", "q", ["x1", "", "x3"], "z", k=2)
    assert r["cus"] == ["x1", "x1", "x3"]
    assert [w["reused"] for w in r["workers"]] == [False, True, False]
    assert r["workers"][1]["cu_in_words"] == 1
    assert r["workers"][1]["cu_out_words"] == 1
    # 首块就空生成：CU_0 本来就是空串，沿用后仍是空
    r = _same(solve, "a b\n\nc d", "q", ["", "x2"], "z", k=2)
    assert r["cus"] == ["", "x2"]
    assert r["workers"][0]["reused"] is True
    assert r["workers"][1]["cu_in_words"] == 0


def check_truncation(solve):
    """max_cu 截断按词；不截断时 CU 原样保留（不归一空格）。"""
    raw = "  keep   two  spaces  here "
    r = _same(solve, "a b", "q", [raw], "z", k=2, max_cu=3)
    assert r["cus"] == ["keep two spaces"]       # 截断走分词 → 归一
    assert r["workers"][0]["truncated"] is True
    assert r["workers"][0]["raw_words"] == 4
    assert r["workers"][0]["cu_out_words"] == 3
    # 不截断：原样返回，前导/连续空格都在
    r = _same(solve, "a b", "q", [raw], "z", k=2)
    assert r["cus"] == [raw]
    assert r["workers"][0]["truncated"] is False
    # 恰好等于上限不截断
    r = _same(solve, "a b", "q", ["one two three"], "z", k=2, max_cu=3)
    assert r["workers"][0]["truncated"] is False
    assert r["cus"] == ["one two three"]


def check_manager_only_last(solve):
    """manager 只吃 CU_l：不喂全部 CU、也不吃原文。"""
    r = _same(solve, _LONG, "what", _W3, "final", k=4)
    assert len(r["manager_cus"]) == 1
    assert r["manager_cus"] == [r["cu_l"]]
    # 输入词数 = query + cu_l，与块数、原文长度都无关
    assert r["manager_input_words"] == 1 + len("CU three four".split())
    # 原文更长（块更多）但 CU_l 相同 → manager 输入一个词都不多
    r2 = _same(solve, _LONG + "\n\nextra words here", "what",
               _W3 + ["CU three four"], "final", k=4)
    assert r2["num_chunks"] == 4
    assert r2["manager_input_words"] == r["manager_input_words"]
    assert r2["cu_l"] == "CU three four"
    # 中间 CU 再长也不进 manager 输入
    r3 = _same(solve, _LONG, "what",
               ["a b c d e f", "g h i j k l", "m"], "final", k=4)
    assert r3["manager_input_words"] == 2
    assert r3["cus"] == ["a b c d e f", "g h i j k l", "m"]


def check_reverse_order(solve):
    """right-to-left 只改处理顺序：块表仍是原文顺序，CU 链跟着走。"""
    r = _same(solve, "a b\n\nc d\n\ne f", "q", ["x1", "x2", "x3"], "z",
              k=2, order="right-to-left")
    assert r["order"] == [2, 1, 0]
    assert [c["text"] for c in r["chunks"]] == ["a b", "c d", "e f"]
    assert [w["chunk_idx"] for w in r["workers"]] == [2, 1, 0]
    assert r["cus"] == ["x3", "x2", "x1"]
    assert r["cu_l"] == "x1"
    assert r["manager_cus"] == ["x1"]
    # 逆序下空生成沿用的是「逆序意义上的上一个」
    r = _same(solve, "a b\n\nc d\n\ne f", "q", ["x1", "", "x3"], "z",
              k=2, order="right-to-left")
    assert r["cus"] == ["x3", "x3", "x1"]
    assert [w["reused"] for w in r["workers"]] == [False, True, False]


def check_window(solve):
    """给了 window 才校验：worker 与 manager 都可能撑爆。"""
    # 不传 window：再长也不管（生成多长都不影响本轮输入，只影响下一轮）
    r = _same(solve, "a b", "q", ["one two three four five"], "z", k=2)
    assert r["workers"][0]["input_words"] == 3
    # 块 2 词 + query 1 词 = 3，window=3 不报错
    r = _same(solve, "a b", "q", ["one two"], "z", k=2, window=3)
    assert r["workers"][0]["input_words"] == 3
    # 同一个用例把 window 收到 2 就撑爆
    _bad(lambda: solve("a b", "q", ["one two"], "z", k=2, window=2))
    # 第二轮才撑爆：CU 2 词 + 块 2 词 + query 1 词 = 5
    _bad(lambda: solve("a b\n\nc d", "q", ["one two", "x"], "z",
                       k=2, window=4))
    # manager 超窗（worker 没超）
    _bad(lambda: solve("a b", "q", ["one two"], "z", k=2, window=3,
                       manager_instr_words=1))
    # instr_words 计入 worker 输入
    _bad(lambda: solve("a b", "q", ["one"], "z", k=2, window=3,
                       instr_words=1))


def check_defaults(solve):
    """默认 k=8、max_cu=None、window=None、指令 0 词、左到右。"""
    src = " ".join("w%d" % i for i in range(10))
    r = _same(solve, src, "q", ["one two", "three four"], "z")
    assert r["num_chunks"] == 2                 # 10 词按 8 切：8 + 2
    assert [c["words"] for c in r["chunks"]] == [8, 2]
    assert r["workers"][0]["truncated"] is False   # max_cu 默认不限
    assert r["order"] == [0, 1]
    assert r["workers"][0]["input_words"] == 0 + 0 + 8 + 1
    # 8 词都能进一块
    r = _same(solve, " ".join("x%d" % i for i in range(8)), "q",
              ["cu"], "z")
    assert r["num_chunks"] == 1
    assert r["manager_input_words"] == 2        # 只有 query + cu


def check_errors(solve):
    """域约束：窗口、块数、空源、非法 order、非法上限。"""
    assert _bad(lambda: solve("a b", "q", ["x"], "z", k=0))
    assert _bad(lambda: solve("a b", "q", ["x"], "z", k=-3))
    assert _bad(lambda: solve("a b", "q", ["x"], "z", k=2, max_cu=0))
    assert _bad(lambda: solve("a b", "q", ["x"], "z", k=2,
                              order="middle-out"))
    # 生成条数与块数不符
    assert _bad(lambda: solve("a b\n\nc d", "q", ["x"], "z", k=2))
    assert _bad(lambda: solve("a b", "q", ["x", "y"], "z", k=2))
    # 空源 / 全空白源都切不出块；给一条生成也一样（别拿 arity 兜底）
    assert _bad(lambda: solve("", "q", [], "z"))
    assert _bad(lambda: solve("   \n\n \t ", "q", [], "z"))
    assert _bad(lambda: solve("", "q", ["x"], "z"))
    assert _bad(lambda: solve("   \n\n \t ", "q", ["x"], "z"))
    # 错误里带的是 window 值而不是 k（k=2、window=1）
    try:
        solve("a b", "q", ["x"], "z", k=2, window=1)
    except ValueError as exc:
        assert "1" in str(exc) and "> 1" in str(exc), str(exc)
    else:
        raise AssertionError("expected ValueError")


def check_random_cross(solve):
    """随机源文本 / 窗口 / 生成长度，与 oracle 全账比对。"""
    seed = 20241102
    for _ in range(30):
        seed = (seed * 1103515245 + 12345) % (1 << 31)
        paras, n_para = [], 1 + seed % 5
        for p in range(n_para):
            seed = (seed * 1103515245 + 12345) % (1 << 31)
            n_word = seed % 6
            paras.append(" ".join("t%d" % (seed % 7) for _ in range(n_word)))
        source = "\n\n".join(paras)
        seed = (seed * 1103515245 + 12345) % (1 << 31)
        k = 1 + seed % 5
        # 先看会切出几块，再按块数造生成
        probe = _bucket(source, k)
        if not probe:
            assert _bad(lambda: solve(source, "q", [], "z", k=k))
            continue
        seed = (seed * 1103515245 + 12345) % (1 << 31)
        cap = [None, 1, 2, 3][seed % 4]
        seed = (seed * 1103515245 + 12345) % (1 << 31)
        rev = seed % 2 == 0
        outs = []
        for i in range(len(probe)):
            seed = (seed * 1103515245 + 12345) % (1 << 31)
            n = seed % 5                      # 0 表示本块无信息
            outs.append(" ".join("c%d" % (seed % 4) for _ in range(n)))
        kw = {"k": k, "max_cu": cap,
              "order": "right-to-left" if rev else "left-to-right"}
        r = _same(solve, source, "q", outs, "ans", **kw)
        assert r["num_chunks"] == len(probe)
        assert sum(c["words"] for c in r["chunks"]) == \
            sum(len(b) for b in probe)
        assert all(c["words"] <= k for c in r["chunks"])


def check_example(solve):
    """题面示例。"""
    r = _same(solve,
              "The cat sat.\n\nThe dog ran fast.\n\nA bird flew away.",
              "who ran", ["cat", "dog ran", "bird flew"], "the dog", k=4)
    assert r["num_chunks"] == 3
    assert r["cus"] == ["cat", "dog ran", "bird flew"]
    assert r["cu_l"] == "bird flew"
    assert r["manager_cus"] == ["bird flew"]
    assert r["answer"] == "the dog"
    # answer 原样返回：manager 生成的前后空白不剥
    r = _same(solve, "The cat sat.", "who", ["cat"], "  the dog  ", k=4)
    assert r["answer"] == "  the dog  "


def _bad(fn):
    try:
        fn()
    except ValueError:
        return True
    except Exception as exc:                       # noqa: BLE001
        raise AssertionError("wrong error: %r" % (exc,))
    raise AssertionError("expected ValueError")


check_example.example = True
