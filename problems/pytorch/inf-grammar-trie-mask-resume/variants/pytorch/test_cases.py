import os


def _py(encoded_vocab, targets, uncertain_token_ids, stop_token_ids=None):
    """独立重写的 oracle：排序用「稳定排序按 token」（并列保持 id 升序）、
    trie 用「top not in tok」等价改写、lcp 用 os.path.commonprefix、
    匹配主循环用 while + 显式状态变量——与 solution 不同代码路径，同官方语义。"""
    DETECT = {
        "</s>", "<|end_of_text|>", "<|eot_id|>", "<|endoftext|>",
        "<eos>", "<|eos|>", "<end_of_turn>", "<｜end▁of▁sentence｜>",
    }
    if not isinstance(encoded_vocab, (list, tuple)) or any(
        not isinstance(t, str) for t in encoded_vocab
    ):
        raise ValueError("bad vocab")
    if not isinstance(targets, (list, tuple)) or any(
        not isinstance(t, str) for t in targets
    ):
        raise ValueError("bad targets")
    if stop_token_ids is not None:
        if isinstance(stop_token_ids, (str, bytes)) or not isinstance(
            stop_token_ids, (list, tuple, set, frozenset)
        ):
            raise ValueError("bad stop ids")
        for i in stop_token_ids:
            if isinstance(i, bool) or not isinstance(i, int) or i < 0 or i >= len(encoded_vocab):
                raise ValueError("bad stop id value")
    if not isinstance(uncertain_token_ids, (list, tuple)):
        raise ValueError("bad uncertain")

    if stop_token_ids is None:
        stop_ids = set()
        for i in range(len(encoded_vocab)):
            if encoded_vocab[i] in DETECT:
                stop_ids.add(i)
    else:
        stop_ids = set()
        for i in stop_token_ids:
            stop_ids.add(int(i))

    rest = []
    special_ids = set()
    for i in range(len(encoded_vocab)):
        if i in stop_ids:
            continue
        tok = encoded_vocab[i]
        if tok == "":
            special_ids.add(i)
        else:
            rest.append(i)
    # 稳定排序按 token：插入序即 id 升序，并列自动保持 id 升序
    rest_sorted = sorted(rest, key=lambda i: encoded_vocab[i])
    sorted_token_ids = rest_sorted
    sorted_tokens = [encoded_vocab[i] for i in rest_sorted]

    # trie：官方单调栈的 quirk 语义（子串包含判断）用 not-in 改写
    m = len(sorted_tokens)
    trie_range = [0] * m
    stk = []
    for i in range(m):
        tok = sorted_tokens[i]
        while stk and stk[-1][0] not in tok:
            j = stk.pop()[1]
            trie_range[j] = i
        stk.append((tok, i))
    while stk:
        j = stk.pop()[1]
        trie_range[j] = m

    id2idx = {tid: k for k, tid in enumerate(sorted_token_ids)}

    u = []
    for tid in uncertain_token_ids:
        if isinstance(tid, bool) or not isinstance(tid, int):
            raise ValueError("bad uncertain id")
        if tid < 0 or tid >= len(encoded_vocab):
            raise ValueError("uncertain id out of range")
        if tid in stop_ids or tid in special_ids:
            raise ValueError("uncertain contains stop/special")
        u.append(id2idx[tid])
    u.sort()

    tgt = list(targets)

    def advance(s):
        for t in tgt:
            if t.startswith(s):
                return True
        return False

    accepted_ids = []
    ledger = []
    prev_tok = None
    prev_matched = 0
    last_rej = 0
    k = 0
    while k < len(u):
        idx = u[k]
        k += 1
        cur = sorted_tokens[idx]
        tid = sorted_token_ids[idx]
        if idx < last_rej:
            ledger.append({"token_id": tid, "sorted_index": idx,
                           "branch": "subtree-skip", "lcp": -1, "matched_from": -1})
            continue
        lcp = -1 if prev_tok is None else len(os.path.commonprefix([cur, prev_tok]))
        accepted = True
        branch = None
        if prev_tok is not None:
            if lcp > prev_matched:
                last_rej = trie_range[idx]
                accepted = False
                branch = "lcp-short-circuit"
            elif lcp < prev_matched:
                prev_matched = lcp
        matched_from = -1
        if accepted:
            matched_from = prev_matched
            j = matched_from
            while j < len(cur):
                if not advance(cur[:j + 1]):
                    last_rej = trie_range[idx]
                    accepted = False
                    branch = "walk-rejected"
                    break
                j += 1
                prev_matched = j
            if accepted:
                branch = "accepted"
                accepted_ids.append(tid)
        ledger.append({"token_id": tid, "sorted_index": idx, "branch": branch,
                       "lcp": lcp, "matched_from": matched_from})
        prev_tok = cur

    return {
        "sorted_token_ids": sorted_token_ids,
        "sorted_tokens": sorted_tokens,
        "trie_subtree_nodes_range": trie_range,
        "stop_token_ids": sorted(stop_ids),
        "special_token_ids": sorted(special_ids),
        "accepted_token_ids": accepted_ids,
        "ledger": ledger,
    }


def _compare(solve, *args, **kwargs):
    led = solve(*args, **kwargs)
    ref = _py(*args, **kwargs)
    assert led == ref, f"\nled={led}\nref={ref}"
    return led


def check_trie_build(solve):
    vocab = ["get", "getter", "go", "gone", "set", "setter", "a", "", "<eos>"]
    led = _compare(solve, vocab, ["get"], [])
    assert led["sorted_tokens"] == ["a", "get", "getter", "go", "gone", "set", "setter"]
    assert led["sorted_token_ids"] == [6, 0, 1, 2, 3, 4, 5]
    # 子树 = [i, range[i])：a→1；get 的子树 [1,3)（getter 是前缀扩展，go 不是）；
    # getter→3；go 的子树 [3,5)（gone）；gone→5；set 的子树 [5,7)（setter）；setter→7
    assert led["trie_subtree_nodes_range"] == [1, 3, 3, 5, 5, 7, 7]
    # 检测表命中 <eos>；空串是 special；"get" 含子串 "a" 但 find 语义正确弹出
    assert led["stop_token_ids"] == [8]
    assert led["special_token_ids"] == [7]


def check_quirk_substring_prefix(solve):
    # 官方 quirk：前缀判断用 find（子串包含）而非 startswith——
    # "xxab" 含子串 "xab" → 栈顶 "xab" 不弹出 → range["xab"] 被错误延长到 2
    vocab = ["xab", "xxab"]
    led = _compare(solve, vocab, ["xa", "xxab"], [])
    assert led["trie_subtree_nodes_range"] == [2, 2]  # 正确语义应为 [1, 2]
    # quirk 的 matcher 后果："xab" walk 失败后其被错误延长的子树把
    # "xxab" 整段 subtree-skip——尽管独立 walk 会接受它（xxab ∈ targets）
    led2 = _compare(solve, vocab, ["xa", "xxab"], [], None)
    led3 = solve(vocab, ["xa", "xxab"], [0, 1])
    assert led3["ledger"][0]["branch"] == "walk-rejected"
    assert led3["ledger"][1]["branch"] == "subtree-skip"
    assert led3["accepted_token_ids"] == []


def check_incremental_resume(solve):
    # get→go 回退续推、goa 失败后 gone 复用其已匹配前缀（lcp==matched 不回退）
    vocab = ["go", "get", "gone", "got", "gox", "goa"]
    led = _compare(solve, vocab, ["get", "go", "gone"], [0, 1, 2, 3, 4, 5])
    assert led["sorted_token_ids"] == [1, 0, 5, 2, 3, 4]
    assert led["trie_subtree_nodes_range"] == [1, 6, 3, 4, 5, 6]
    assert led["accepted_token_ids"] == [1, 0, 2]
    br = [e["branch"] for e in led["ledger"]]
    assert br == ["accepted", "accepted", "walk-rejected", "accepted",
                  "walk-rejected", "walk-rejected"]
    # get 从 0 起推；go 与 get 的 lcp=1（g 相同、e≠o）→ 回退到 1 续推；
    # goa 失败后 gone 复用其已匹配前缀 2（lcp==matched 不回退）
    mf = [e["matched_from"] for e in led["ledger"]]
    assert mf == [0, 1, 2, 2, 2, 2]
    lcp = [e["lcp"] for e in led["ledger"]]
    assert lcp == [-1, 1, 2, 2, 2, 2]


def check_lcp_short_circuit(solve):
    # abxd 在字节 2 失败（prev_matched 停在 2）；abxyc 与它的 lcp=3 > 2
    # 但不以其为前缀（子树外）→ lcp 短路拒；短路同时置 last_rejected_range，
    # 使 abxycd（其子树内）被 subtree-skip；aby 复用前缀 2 接受
    vocab = ["ab", "abxd", "abxyc", "abxycd", "aby"]
    led = _compare(solve, vocab, ["ab", "aby"], [0, 1, 2, 3, 4])
    assert led["trie_subtree_nodes_range"] == [5, 2, 4, 4, 5]
    assert led["accepted_token_ids"] == [0, 4]
    br = [e["branch"] for e in led["ledger"]]
    assert br == ["accepted", "walk-rejected", "lcp-short-circuit",
                  "subtree-skip", "accepted"]
    assert led["ledger"][2]["lcp"] == 3
    assert led["ledger"][2]["matched_from"] == -1
    assert led["ledger"][4]["matched_from"] == 2


def check_subtree_skip(solve):
    # "abc" 失败后其整棵子树（abcd、abcde）全部 subtree-skip
    vocab = ["ab", "abc", "abcd", "abcde"]
    led = _compare(solve, vocab, ["ab"], [0, 1, 2, 3])
    assert led["trie_subtree_nodes_range"] == [4, 4, 4, 4]
    assert led["accepted_token_ids"] == [0]
    br = [e["branch"] for e in led["ledger"]]
    assert br == ["accepted", "walk-rejected", "subtree-skip", "subtree-skip"]
    # range 值边界：range["ab"]=1 恰好不含 "ac"（不以 ab 为前缀）→
    # ab 失败后 ac 不被 skip，独立 walk 接受
    led2 = _compare(solve, ["ab", "ac"], ["ac"], [0, 1])
    assert led2["trie_subtree_nodes_range"] == [1, 2]
    assert led2["accepted_token_ids"] == [1]
    assert [e["branch"] for e in led2["ledger"]] == ["walk-rejected", "accepted"]


def check_stop_and_special(solve):
    # 检测表模式（stop_token_ids=None）：官方 DETECTION_STOP_TOKENS 8 条
    vocab = ["</s>", "hello", "", "<eos>", " ", "<|eot_id|>", "help"]
    led = _compare(solve, vocab, ["get"], [])
    assert led["stop_token_ids"] == [0, 3, 5]
    assert led["special_token_ids"] == [2]  # 仅空串；" " 不是 special
    assert led["sorted_tokens"] == [" ", "hello", "help"]
    # 显式模式：不再用检测表，"</s>"/"<eos>"/"<|eot_id|>" 全部进 sorted
    led2 = _compare(solve, vocab, ["get"], [], [1])
    assert led2["stop_token_ids"] == [1]
    assert led2["special_token_ids"] == [2]
    assert led2["sorted_tokens"] == [" ", "</s>", "<eos>", "<|eot_id|>", "help"]
    assert led2["sorted_token_ids"] == [4, 0, 3, 5, 6]


def check_sort_tie_id(solve):
    # 相同 token 字符串并列：官方 std::sort 不稳定，题面钉死按 id 升序
    vocab = ["go", "get", "go"]
    led = _compare(solve, vocab, ["go"], [0, 1, 2])
    assert led["sorted_token_ids"] == [1, 0, 2]
    assert led["trie_subtree_nodes_range"] == [1, 3, 3]
    # get 在字节 1 失败；go(0) 复用前缀 1；go(2) 与 go(0) 的 lcp=2==matched
    # 且已完全匹配（无剩余字节）→ 直接 accepted
    assert led["accepted_token_ids"] == [0, 2]
    br = [e["branch"] for e in led["ledger"]]
    assert br == ["walk-rejected", "accepted", "accepted"]
    assert [e["matched_from"] for e in led["ledger"]] == [0, 1, 2]


def check_independent_walk_equivalence(solve):
    # 无 quirk vocab 上：增量复用算法的 accepted 集合必须等于
    # 每 token 从零独立 walk 的真值
    def indep(vocab, targets, uncert):
        out = []
        for tid in uncert:
            tok = vocab[tid]
            ok = True
            for j in range(len(tok)):
                if not any(t.startswith(tok[:j + 1]) for t in targets):
                    ok = False
                    break
            if ok:
                out.append(tid)
        return out

    cases = [
        (["go", "get", "gone", "got", "gox", "goa"], ["get", "go", "gone"],
         [0, 1, 2, 3, 4, 5]),
        (["ab", "abxd", "abxyc", "aby"], ["ab", "aby"], [0, 1, 2, 3]),
        (["ab", "abc", "abcd", "abcde"], ["ab"], [0, 1, 2, 3]),
        (["x1", "x2", "y1", "y22", "y"], ["y2", "x"], [0, 1, 2, 3, 4]),
    ]
    for vocab, targets, uncert in cases:
        led = _compare(solve, vocab, targets, uncert)
        assert sorted(led["accepted_token_ids"]) == indep(vocab, targets, uncert)


def check_empty_targets_and_edges(solve):
    # targets 为空：第一 token 在字节 0 失败；后续要么 lcp 短路要么同样失败
    led = _compare(solve, ["ab", "cd"], [], [0, 1])
    assert led["accepted_token_ids"] == []
    assert [e["branch"] for e in led["ledger"]] == ["walk-rejected", "walk-rejected"]
    # uncertain 为空
    led2 = _compare(solve, ["ab"], ["ab"], [])
    assert led2["accepted_token_ids"] == [] and led2["ledger"] == []
    # 空 vocab
    led3 = _compare(solve, [], [], [])
    assert led3["sorted_token_ids"] == [] and led3["trie_subtree_nodes_range"] == []


def check_validation(solve):
    bad = [
        lambda: solve("vocab", [], []),
        lambda: solve([1, 2], [], []),
        lambda: solve(["a"], [1], []),
        lambda: solve(["a"], [], [], stop_token_ids="0"),
        lambda: solve(["a"], [], [], stop_token_ids=[True]),
        lambda: solve(["a"], [], [], stop_token_ids=[5]),
        lambda: solve(["a"], [], [True]),
        lambda: solve(["a"], [], [3]),
        lambda: solve(["a"], [], ["0"]),
    ]
    for fn in bad:
        try:
            fn()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")
    # uncertain 含 stop / special token → ValueError
    try:
        solve(["</s>", "ab"], ["ab"], [0, 1])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
    try:
        solve(["", "ab"], ["ab"], [0])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
