"""order 835 判题用例：独立 oracle 用纯 Python 重写，不 import solution。

3 个 check_*：split_segments / remove_long_silence / mel_length_ledger。
每个都做全网格对拍 + 手算锚点（手算锚点独立于 oracle 与 solution）。
"""

PUNCT = [".", "!", "?", "▁.", "▁?", "▁..."]


# ---------------------------------------------------------------- oracle
def _ora_split_impl(tokens, split_tokens, max_seg, quick):
    if len(tokens) == 0:
        return [], [], 0, []
    segs = []
    cur = []
    L = 0
    for i in range(len(tokens)):
        t = tokens[i]
        cur.append(t)
        L += 1
        if not ("," in split_tokens or "▁," in split_tokens) and \
                ("," in cur or "▁," in cur):
            sub = _ora_split_impl(cur, [",", "▁,"], max_seg, quick)[0]
        elif "-" not in split_tokens and "-" in cur:
            sub = _ora_split_impl(cur, ["-"], max_seg, quick)[0]
        elif L <= max_seg:
            if t in split_tokens and L > 2:
                if i < len(tokens) - 1 and tokens[i + 1] in ["'", "▁'"]:
                    cur.append(tokens[i + 1])
                    i += 1
                segs.append(cur)
                cur, L = [], 0
            continue
        else:
            sub = []
            for j in range(0, len(cur), max_seg):
                sub.append(cur[j:j + max_seg]
                           if j + max_seg < len(cur) else cur[j:])
        segs.extend(sub)
        cur, L = [], 0
    if L > 0:
        assert L <= max_seg
        segs.append(cur)

    merged, total, flags = [], 0, []
    for s in segs:
        total += len(s)
        if len(s) == 0:
            continue
        if len(merged) == 0:
            merged.append(s)
            flags.append("first")
        elif len(merged[-1]) + len(s) <= max_seg and total > quick:
            merged[-1] = merged[-1] + s
            flags.append("merge_quick")
        elif len(merged[-1]) + len(s) <= max_seg / 2:
            merged[-1] = merged[-1] + s
            flags.append("merge_half")
        else:
            merged.append(s)
            flags.append("new")
    return merged, segs, total, flags


def ora_split(tokens, max_seg, split_tokens, quick):
    merged, pre, total, flags = _ora_split_impl(
        list(tokens), list(split_tokens), max_seg, quick)
    return {"segments": merged, "pre_merge": pre, "total_token": total,
            "n_segments": len(merged), "merge_flags": flags}


def ora_remove(codes, silent, stop, maxc):
    code_lens, codes_list, counts, kept = [], [], [], []
    isfix = False
    for code in codes:
        len_ = code.index(stop) if stop in code else len(code)
        count = sum(1 for v in code if v == silent)
        counts.append(count)
        if count > maxc:
            idxs, n = [], 0
            for k in range(len_):
                assert code[k] != stop
                if code[k] != silent:
                    idxs.append(k)
                    n = 0
                elif n < 10:
                    idxs.append(k)
                    n += 1
            len_ = len(idxs)
            codes_list.append([code[k] for k in idxs])
            kept.append(idxs)
            isfix = True
        else:
            codes_list.append(code[:len_])
            kept.append(list(range(len_)))
        code_lens.append(len_)
    T = len(codes[0]) if codes else 0
    if isfix:
        L = max(len(c) for c in codes_list)
        out = [c + [stop] * (L - len(c)) for c in codes_list]
        rebuilt = True
    else:
        out = [list(c) for c in codes]
        rebuilt = False
    max_len = max(code_lens) if code_lens else 0
    if max_len < T:
        out = [c[:max_len] for c in out]
    return {"codes": out, "code_lens": code_lens, "isfix": isfix,
            "counts": counts, "kept_indices": kept, "rebuilt": rebuilt,
            "max_len": max_len}


def ora_mel(codes, stop, ratio, max_mel):
    has_warned = any(len(c) > 0 and c[-1] != stop for c in codes) \
        if codes else False
    cls = [c.index(stop) if stop in c else len(c) for c in codes]
    m = max(cls) if cls else 0
    trunc = [c[:m] for c in codes]
    tgt = [int(cl * ratio) for cl in cls]
    return {"code_lens": cls, "max_code_len": m,
            "truncated_codes": trunc, "has_warned": has_warned,
            "target_lengths": tgt, "total_frames": sum(tgt),
            "ratio": ratio, "max_mel_tokens": max_mel}


# ---------------------------------------------------------------- checks
def check_split_segments(solve):
    toks = [
        [],
        ["a", "b", "c", "."],
        ["a", ",", "b", ".", "c"],
        ["a", "-", "b", ".", "c"],
        ["a", "b", ".", "'", "z"],
        ["x"] * 7,
        ["▁a", "▁b", "▁,", "▁c", "▁."],
        ["."],
        ["a", "b", "!", "c", "?", "d"],
        ["a", "b", "▁'", "c", "▁.", "d"],
        ["a", "b", "c", "-"],
    ]
    max_segs = (2, 3, 120)
    quicks = (0, 3, 10)
    splits = (PUNCT, ["."], ["!", "?"], [","])
    n = 0
    for t in toks:
        for ms in max_segs:
            for q in quicks:
                for sp in splits:
                    got = solve(mode="split_segments", tokenized=t,
                                max_text_tokens_per_segment=ms,
                                split_tokens=sp, quick_streaming_tokens=q)
                    exp = ora_split(t, ms, sp, q)
                    assert got == exp, ("split t=%r ms=%r q=%r sp=%r\n got=%r\n exp=%r"
                                        % (t, ms, q, sp, got, exp))
                    n += 1
    # 默认 split_tokens（None）必须等价于 punctuation_marks_tokens；
    # 该序列的 "!" 位于 cur_len=4，删掉默认常量中的 "!" 会漏掉这个切分点。
    t = ["a", "b", "c", "!", "d", "e"]
    assert solve(mode="split_segments", tokenized=t,
                 max_text_tokens_per_segment=120,
                 quick_streaming_tokens=0) == ora_split(t, 120, PUNCT, 0)
    # 手算锚点 1：空输入
    assert solve(mode="split_segments", tokenized=[],
                 max_text_tokens_per_segment=120, split_tokens=["."],
                 quick_streaming_tokens=0) == {
        "segments": [], "pre_merge": [], "total_token": 0,
        "n_segments": 0, "merge_flags": []}
    # 手算锚点 2：单段标点收尾
    assert solve(mode="split_segments", tokenized=["a", "b", "c", "."],
                 max_text_tokens_per_segment=120, split_tokens=["."],
                 quick_streaming_tokens=0) == {
        "segments": [["a", "b", "c", "."]], "pre_merge": [["a", "b", "c", "."]],
        "total_token": 4, "n_segments": 1, "merge_flags": ["first"]}
    # 手算锚点 3：' 吞并 + i+=1 不生效（' 双重出现）+ 合并成一段
    assert solve(mode="split_segments", tokenized=["a", "b", ".", "'", "z"],
                 max_text_tokens_per_segment=120, split_tokens=["."],
                 quick_streaming_tokens=0) == {
        "segments": [["a", "b", ".", "'", "'", "z"]],
        "pre_merge": [["a", "b", ".", "'"], ["'", "z"]],
        "total_token": 6, "n_segments": 1,
        "merge_flags": ["first", "merge_quick"]}
    # 手算锚点 4：逗号递归
    assert solve(mode="split_segments", tokenized=["a", ",", "b"],
                 max_text_tokens_per_segment=120, split_tokens=["."],
                 quick_streaming_tokens=0) == {
        "segments": [["a", ",", "b"]], "pre_merge": [["a", ","], ["b"]],
        "total_token": 3, "n_segments": 1,
        "merge_flags": ["first", "merge_quick"]}
    # 手算锚点 5：超长硬切 + max/2 浮点条件
    assert solve(mode="split_segments", tokenized=["x"] * 7,
                 max_text_tokens_per_segment=3, split_tokens=["."],
                 quick_streaming_tokens=0) == {
        "segments": [["x", "x", "x"], ["x"], ["x", "x", "x"]],
        "pre_merge": [["x", "x", "x"], ["x"], ["x", "x", "x"]],
        "total_token": 7, "n_segments": 3,
        "merge_flags": ["first", "new", "new"]}
    # 手算锚点 6：默认 split_tokens 含 "!" 且真正触发切分（默认常量不得缺项）
    assert solve(mode="split_segments",
                 tokenized=["a", "b", "c", "!", "d", "e"],
                 max_text_tokens_per_segment=120,
                 quick_streaming_tokens=0) == {
        "segments": [["a", "b", "c", "!", "d", "e"]],
        "pre_merge": [["a", "b", "c", "!"], ["d", "e"]],
        "total_token": 6, "n_segments": 1,
        "merge_flags": ["first", "merge_quick"]}
    # 手算锚点 7：连字符递归遇超长 cur 时内部硬切成两段（递归不可退化为整段）
    assert solve(mode="split_segments", tokenized=["a", "b", "c", "-"],
                 max_text_tokens_per_segment=3, split_tokens=["."],
                 quick_streaming_tokens=0) == {
        "segments": [["a", "b", "c"], ["-"]],
        "pre_merge": [["a", "b", "c"], ["-"]],
        "total_token": 4, "n_segments": 2,
        "merge_flags": ["first", "new"]}
    print("split_segments: %d grid + 7 anchors OK" % n)


def check_remove_long_silence(solve):
    codes = [
        [[1, 2, 3, 4], [5, 6, 7, 8]],
        [[7, 999, 52, 52], [1, 999, 52, 52]],
        [[52, 52, 52, 52], [1, 2, 3, 4]],
        [[52] * 20, [1] * 20],
        [[7, 999] + [52] * 4 + [1, 2], [3, 999] + [52] * 4 + [5, 6]],
        [[1, 2, 3, 4], [5, 6, 7, 8]],
    ]
    params = [
        (52, 999, 30), (52, 999, 3), (1, 999, 0), (52, 8, 3),
    ]
    n = 0
    for c in codes:
        for silent, stop, maxc in params:
            got = solve(mode="remove_long_silence", codes=c,
                        silent_token=silent, stop_mel_token=stop,
                        max_consecutive=maxc)
            exp = ora_remove(c, silent, stop, maxc)
            assert got == exp, ("remove c=%r p=%r\n got=%r\n exp=%r"
                                % (c, (silent, stop, maxc), got, exp))
            n += 1
    # 手算锚点 1：无 stop / 无 silent / 不收缩 / 不 clip
    assert solve(mode="remove_long_silence",
                 codes=[[1, 2, 3, 4], [5, 6, 7, 8]], silent_token=52,
                 stop_mel_token=999, max_consecutive=30) == {
        "codes": [[1, 2, 3, 4], [5, 6, 7, 8]], "code_lens": [4, 4],
        "isfix": False, "counts": [0, 0],
        "kept_indices": [[0, 1, 2, 3], [0, 1, 2, 3]],
        "rebuilt": False, "max_len": 4}
    # 手算锚点 2：count 含 stop 之后 + isfix=False 仍 clip 到 max_len=1
    assert solve(mode="remove_long_silence",
                 codes=[[7, 999, 52, 52], [1, 999, 52, 52]], silent_token=52,
                 stop_mel_token=999, max_consecutive=30) == {
        "codes": [[7], [1]], "code_lens": [1, 1], "isfix": False,
        "counts": [2, 2], "kept_indices": [[0], [0]],
        "rebuilt": False, "max_len": 1}
    # 手算锚点 3：连续静音最多留 10 + batch 级 isfix + pad 补齐
    assert solve(mode="remove_long_silence",
                 codes=[[52] * 20, [1] * 20], silent_token=52,
                 stop_mel_token=999, max_consecutive=3) == {
        "codes": [[52] * 10 + [999] * 10, [1] * 20],
        "code_lens": [10, 20], "isfix": True, "counts": [20, 0],
        "kept_indices": [list(range(10)), list(range(20))],
        "rebuilt": True, "max_len": 20}
    # 手算锚点 4：stop 之后的 silent 计入 count，但收缩只在 len_ 内
    assert solve(mode="remove_long_silence",
                 codes=[[7, 999] + [52] * 4 + [1, 2],
                        [3, 999] + [52] * 4 + [5, 6]],
                 silent_token=52, stop_mel_token=999, max_consecutive=3) == {
        "codes": [[7], [3]], "code_lens": [1, 1], "isfix": True,
        "counts": [4, 4], "kept_indices": [[0], [0]],
        "rebuilt": True, "max_len": 1}
    # 手算锚点 5：连续静音段被非静音打断时 n 必须归零（否则后续段被吞）
    seg = [52] * 5 + [7] + [52] * 5 + [7] + [52] * 5
    assert solve(mode="remove_long_silence", codes=[seg],
                 silent_token=52, stop_mel_token=999, max_consecutive=3) == {
        "codes": [seg], "code_lens": [17], "isfix": True,
        "counts": [15], "kept_indices": [list(range(17))],
        "rebuilt": True, "max_len": 17}
    print("remove_long_silence: %d grid + 5 anchors OK" % n)


def check_mel_length_ledger(solve):
    codes = [
        [[3, 4, 999, 8], [5, 6, 7]],
        [[999], [999]],
        [[3, 999, 9], [5, 999, 9]],
        [[1, 2, 3, 4], [5, 6, 7, 8]],
        [[7, 999, 2, 2], [3, 4, 5, 6]],
    ]
    params = [
        (999, 1.72, 1500), (999, 2.5, 10), (8, 0.5, 1500),
    ]
    n = 0
    for c in codes:
        for stop, ratio, max_mel in params:
            got = solve(mode="mel_length_ledger", codes=c,
                        stop_mel_token=stop, ratio=ratio,
                        max_mel_tokens=max_mel)
            exp = ora_mel(c, stop, ratio, max_mel)
            assert got == exp, ("mel c=%r p=%r\n got=%r\n exp=%r"
                                % (c, (stop, ratio, max_mel), got, exp))
            n += 1
    # 手算锚点 1：截到 stop + 批内切齐 + has_warned + 1.72 截断 + total
    assert solve(mode="mel_length_ledger", codes=[[3, 4, 999, 8], [5, 6, 7]],
                 stop_mel_token=999, ratio=1.72, max_mel_tokens=1500) == {
        "code_lens": [2, 3], "max_code_len": 3,
        "truncated_codes": [[3, 4, 999], [5, 6, 7]],
        "has_warned": True, "target_lengths": [3, 5], "total_frames": 8,
        "ratio": 1.72, "max_mel_tokens": 1500}
    # 手算锚点 2：stop 在 idx0 -> 空 code_lens / 无 warn / 空 truncated
    assert solve(mode="mel_length_ledger", codes=[[999], [999]],
                 stop_mel_token=999, ratio=1.72, max_mel_tokens=1500) == {
        "code_lens": [0, 0], "max_code_len": 0,
        "truncated_codes": [[], []], "has_warned": False,
        "target_lengths": [0, 0], "total_frames": 0,
        "ratio": 1.72, "max_mel_tokens": 1500}
    # 手算锚点 3：int 截断 vs round 分叉（1*1.72 -> 1 而非 2）
    assert solve(mode="mel_length_ledger", codes=[[3, 999, 9], [5, 999, 9]],
                 stop_mel_token=999, ratio=1.72, max_mel_tokens=1500)[
        "target_lengths"] == [1, 1]
    print("mel_length_ledger: %d grid + 3 anchors OK" % n)


def check(solve):
    check_split_segments(solve)
    check_remove_long_silence(solve)
    check_mel_length_ledger(solve)
    print("all checks passed")
