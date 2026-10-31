def _oracle(text, vocab, unk_token, continuation_prefix, max_input_chars_per_word):
    """结构独立：text.split() 切词 + str.find 定位词区间 + 按首字符分桶、按长度降序的
    str.startswith 命中。参考解是「端点截串 + 词表 dict 查表」，两条路径不共用中间结果。
    """
    index = {}
    for position, entry in enumerate(vocab):
        index.setdefault(entry, position)
    initial = {}
    contending = {}
    for entry in vocab:
        initial.setdefault(entry[0], []).append((len(entry), entry))
        if entry.startswith(continuation_prefix):
            rest = entry[len(continuation_prefix):]
            if rest:
                contending.setdefault(rest[0], []).append((len(rest), entry))
    for bucket in initial.values():
        bucket.sort(reverse=True)
    for bucket in contending.values():
        bucket.sort(reverse=True)

    tokens, ids, offsets, word_ids = [], [], [], []
    cursor = 0
    for word_no, word in enumerate(text.split()):
        begin = text.find(word, cursor)
        cursor = begin + len(word)
        size = len(word)
        local = []
        if size > max_input_chars_per_word:
            local = None
        else:
            start = 0
            while start < size:
                bucket = initial.get(word[start], []) if start == 0 else contending.get(word[start], [])
                hit_length, hit_piece = 0, None
                for length, entry in bucket:
                    probe = entry if start == 0 else entry[len(continuation_prefix):]
                    if length <= hit_length:
                        continue
                    if word.startswith(probe, start):
                        hit_length, hit_piece = length, entry
                        break
                if hit_piece is None:
                    local = None
                    break
                local.append((hit_piece, start, start + hit_length))
                start += hit_length

        if local is None:
            tokens.append(unk_token)
            ids.append(index[unk_token])
            offsets.append([begin, begin + size])
            word_ids.append(word_no)
            continue
        for piece, piece_start, piece_end in local:
            tokens.append(piece)
            ids.append(index[piece])
            offsets.append([begin + piece_start, begin + piece_end])
            word_ids.append(word_no)
    return tokens, ids, offsets, word_ids


EX_VOCAB = ["[UNK]", "the", "un", "##af", "##fable", "bow", "##els", "a", "##a",
            "ha", "##sh", "cat", "##t"]
EX_TEXT = "the unaffable bowels hashcat cat"
EX_FROZEN = (["the", "un", "##af", "##fable", "bow", "##els", "[UNK]", "cat"],
             [1, 2, 3, 4, 5, 6, 0, 11],
             [[0, 3], [4, 6], [6, 8], [8, 13], [14, 17], [17, 20], [21, 28], [29, 32]],
             [0, 1, 1, 1, 2, 2, 3, 4])


def _verify(solve, args, frozen=None):
    text, vocab, unk_token, prefix, max_chars = args
    vocab_snapshot = list(vocab)
    expected = _oracle(*args)
    if frozen is not None:
        assert expected == frozen, ("oracle disagrees with the hand-computed fixture", expected, frozen)
    result = solve(*args)
    assert isinstance(result, tuple) and len(result) == 4, f"expected a 4-tuple, got {type(result)}"
    tokens, ids, offsets, word_ids = result
    assert tokens == expected[0], ("tokens", tokens, expected[0])
    assert ids == expected[1], ("ids", ids, expected[1])
    assert offsets == expected[2], ("offsets", offsets, expected[2])
    assert word_ids == expected[3], ("word_ids", word_ids, expected[3])
    assert len(tokens) == len(ids) == len(offsets) == len(word_ids)
    assert all(type(item) is str for item in tokens)
    assert all(type(item) is int and not isinstance(item, bool) for item in ids + word_ids)
    assert all(type(item) is list and len(item) == 2
               and all(type(x) is int and not isinstance(x, bool) for x in item) for item in offsets)
    position = {entry: i for i, entry in enumerate(vocab)}
    assert ids == [position[item] for item in tokens]
    assert word_ids == sorted(word_ids)
    words = text.split()
    spans, scan = [], 0
    for word in words:
        begin = text.find(word, scan)
        spans.append((begin, begin + len(word)))
        scan = begin + len(word)
    for word_no in set(word_ids):
        group = [(offsets[i], tokens[i]) for i in range(len(tokens)) if word_ids[i] == word_no]
        if len(group) == 1 and group[0][1] == unk_token:
            # 整词 UNK：偏移必须正好覆盖这个词
            assert group[0][0] == [spans[word_no][0], spans[word_no][1]], (text, group, word_no)
            continue
        cursor = group[0][0][0]
        total = 0
        for index_in_group, ((start, end), piece) in enumerate(group):
            assert start == cursor, (text, group)
            assert end > start, (text, group)
            assert text[start:end] == (piece if index_in_group == 0 else piece[len(prefix):]), (
                text, group, prefix)
            cursor = end
            total += end - start
        assert total == len(words[word_no]), (text, group)
    assert vocab == vocab_snapshot, "input mutation"
    return result


def check_example(solve):
    _verify(solve, (EX_TEXT, EX_VOCAB, "[UNK]", "##", 10), EX_FROZEN)
    # 词长上限改成 3：只有 the 与 cat 能切，其余三个词整词 UNK
    _verify(solve, (EX_TEXT, EX_VOCAB, "[UNK]", "##", 3), (
        ["the", "[UNK]", "[UNK]", "[UNK]", "cat"], [1, 0, 0, 0, 11],
        [[0, 3], [4, 13], [14, 20], [21, 28], [29, 32]], [0, 1, 2, 3, 4]))
    _verify(solve, ("hashcat", EX_VOCAB, "[UNK]", "##", 10), (
        ["[UNK]"], [0], [[0, 7]], [0]))
    _verify(solve, ("bowels", EX_VOCAB, "[UNK]", "##", 10), (
        ["bow", "##els"], [5, 6], [[0, 3], [3, 6]], [0, 0]))
    _verify(solve, ("unaffable", EX_VOCAB, "[UNK]", "##", 10), (
        ["un", "##af", "##fable"], [2, 3, 4], [[0, 2], [2, 4], [4, 9]], [0, 0, 0]))


check_example.example = True


def _walk_vocab():
    return ["[UNK]", "walk", "##ing", "##ed", "er", "##er", "cat", "##cat", "the", "a", "##a"]


def check_matches_oracle(solve):
    walk = _walk_vocab()
    cases = [
        ("walked walking walker", walk, "[UNK]", "##", 10),
        ("walk walkwalk a a a", walk, "[UNK]", "##", 4),
        ("the cat walked", walk, "[UNK]", "", 10),
        ("the cat walked", walk, "[UNK]", "##", 1),
        ("catthe", walk, "[UNK]", "##", 10),
        ("  the\tcat  \n walked ", walk, "[UNK]", "##", 10),
        ("the\u00a0cat", walk, "[UNK]", "##", 10),
        (EX_TEXT, EX_VOCAB, "[UNK]", "##", 10),
        (EX_TEXT, EX_VOCAB, "[UNK]", "##", 5),
        (EX_TEXT, EX_VOCAB, "[UNK]", "", 10),
        (EX_TEXT, EX_VOCAB, "[UNK]", "###", 10),
        ("unaffable unaffable un", EX_VOCAB, "[UNK]", "##", 9),
        ("hashcat cat cat a the bowels", EX_VOCAB, "[UNK]", "##", 10),
        ("", EX_VOCAB, "[UNK]", "##", 10),
        ("   ", EX_VOCAB, "[UNK]", "##", 10),
        ("abcdefgabcdefg", ["[UNK]", "abc", "##defg", "##abcdefg", "a"], "[UNK]", "##", 20),
        ("流行 流行 流", ["[UNK]", "流", "行", "流行", "##行", "##流"], "[UNK]", "##", 4),
        ("\u00e9x \u00e9\u00e9x", ["[UNK]", "\u00e9", "##\u00e9", "x", "##x"], "[UNK]", "##", 6),
        ("a" * 40, ["[UNK]", "a", "##a"], "[UNK]", "##", 39),
        ("a" * 40, ["[UNK]", "a", "##a"], "[UNK]", "##", 40),
        ("aa aaaa aa", ["[UNK]", "a", "##a", "##aa"], "[UNK]", "##", 4),
        ("[UNK] [UNK]a", EX_VOCAB, "[UNK]", "##", 8),
        # unk_token 不在词表首位，id 必须走真实下标
        ("hashcat", ["a", "the", "[UNK]", "##t"], "[UNK]", "##", 10),
        ("the hashcat", ["a", "the", "[UNK]", "##t"], "[UNK]", "##", 10),
    ]
    for args in cases:
        _verify(solve, args)


def check_word_level_unk_drops_pieces(solve):
    # 词头两个片段本可命中，第三位无命中 -> 整词只有一个 UNK，已切片段全部作废
    tokens, ids, offsets, word_ids = _verify(
        solve, ("hashcat", EX_VOCAB, "[UNK]", "##", 10))
    assert tokens == ["[UNK]"] and ids == [0] and offsets == [[0, 7]] and word_ids == [0]
    # 多词时一个词的 UNK 不污染邻居
    tokens, ids, offsets, word_ids = _verify(
        solve, ("the hashcat bowels cat", EX_VOCAB, "[UNK]", "##", 10))
    assert tokens == ["the", "[UNK]", "bow", "##els", "cat"], tokens
    assert offsets == [[0, 3], [4, 11], [12, 15], [15, 18], [19, 22]], offsets
    assert word_ids == [0, 1, 2, 2, 3], word_ids
    # 逐字 UNK 的答法会输出若干片段 + 词中 UNK；这里断言任何 UNK 都独占一个词
    vocab = ["[UNK]", "ha", "##sh", "bow", "##els", "un", "##af", "the", "abc", "##c"]
    for text in ("hashcat bowels", "unaffable the", "abc hashcat", "ha ha ha"):
        result = _verify(solve, (text, vocab, "[UNK]", "##", 20))
        pieces, spans, owners = result[0], result[2], result[3]
        for i, piece in enumerate(pieces):
            if piece != "[UNK]":
                continue
            same_word = [j for j in range(len(pieces)) if owners[j] == owners[i]]
            assert len(same_word) == 1, (text, pieces, owners)
            word = text.split()[owners[i]]
            assert spans[i][1] - spans[i][0] == len(word), (text, i, spans[i], word)


def check_greedy_is_not_minimum_pieces(solve):
    # 贪心在 start=0 取更长的 ab，于是切成 3 段；存在 2 段的切法但贪心不会走
    greedy_vocab = ["[UNK]", "ab", "##c", "##d", "a", "##bcd"]
    tokens, ids, offsets, word_ids = _verify(solve, ("abcd", greedy_vocab, "[UNK]", "##", 10))
    assert tokens == ["ab", "##c", "##d"], tokens
    assert ids == [1, 2, 3] and offsets == [[0, 2], [2, 3], [3, 4]] and word_ids == [0, 0, 0]
    assert "##bcd" not in tokens
    # 反过来：本可两段的词，贪心在第一格就命中整词
    tokens, ids, offsets, word_ids = _verify(
        solve, ("abcd abcd", ["[UNK]", "abcd", "ab", "##cd"], "[UNK]", "##", 10))
    assert tokens == ["abcd", "abcd"] and offsets == [[0, 4], [5, 9]], tokens
    # 贪心失败时，即便「回溯一下就能切成」也仍然整词 UNK
    tokens, _, _, _ = _verify(
        solve, ("abcz", ["[UNK]", "abc", "ab", "##cz", "##z", "##c"], "[UNK]", "##", 10))
    assert tokens == ["abc", "##z"], tokens
    tokens, _, _, _ = _verify(
        solve, ("abqz", ["[UNK]", "ab", "##q", "a", "##qz"], "[UNK]", "##", 10))
    assert tokens == ["ab", "##qz"], tokens
    # 贪心先取 ab、再取 ##c，第三位无命中 -> 整词 UNK；而 a + ##bcd 本可完整覆盖
    tokens, _, _, _ = _verify(
        solve, ("abcd", ["[UNK]", "ab", "##c", "a", "##bcd"], "[UNK]", "##", 10))
    assert tokens == ["[UNK]"], tokens


def check_offsets_are_code_points(solve):
    # 组合记号：e + U+0301 是两个码点；偏移与词长都按码点算
    combo = ["[UNK]", "\u00e9", "##\u00e9", "x", "##x", "e\u0301", "##\u0301"]
    tokens, ids, offsets, word_ids = _verify(solve, ("\u00e9x e\u0301x", combo, "[UNK]", "##", 6))
    assert offsets[0] == [0, 1] and offsets[1] == [1, 2], offsets
    # 增补平面字符：一个 emoji 是 1 个码点（UTF-16 计数会是 2 个单元）
    astral = ["[UNK]", "\U0001F600", "##\U0001F600"]
    tokens, ids, offsets, word_ids = _verify(
        solve, ("\U0001F600\U0001F600\U0001F600", astral, "[UNK]", "##", 3))
    assert tokens == ["\U0001F600", "##\U0001F600", "##\U0001F600"], tokens
    assert offsets == [[0, 1], [1, 2], [2, 3]], offsets
    assert ids == [1, 2, 2]
    # 上限按码点算刚好放行；按 UTF-16 单元或 UTF-8 字节计数会误判成超长
    tokens, _, offsets, _ = _verify(
        solve, ("\U0001F600x", astral + ["x", "##x"], "[UNK]", "##", 2))
    assert tokens == ["\U0001F600", "##x"] and offsets == [[0, 1], [1, 2]], tokens
    # CJK：3 个字节 = 1 个码点
    cjk = ["[UNK]", "\u6d41", "##\u884c"]
    assert _verify(solve, ("\u6d41\u884c", cjk, "[UNK]", "##", 2))[1] == [1, 2]


def check_prefix_forms(solve):
    # 前缀为空串时，词中间也能用光杆词条
    bare = ["[UNK]", "ab", "bc", "c", "a"]
    tokens, ids, offsets, word_ids = _verify(solve, ("abbc", bare, "[UNK]", "", 10))
    assert tokens == ["ab", "bc"] and offsets == [[0, 2], [2, 4]], tokens
    tokens, ids, offsets, word_ids = _verify(solve, ("abc", bare, "[UNK]", "", 10))
    assert tokens == ["ab", "c"], tokens
    tokens, ids, offsets, word_ids = _verify(solve, ("abc", bare, "[UNK]", "##", 10))
    assert tokens == ["[UNK]"], tokens
    # 多字符前缀
    thick = ["[UNK]", "x", "###x", "yy", "###yy"]
    assert _verify(solve, ("xxyy", thick, "[UNK]", "###", 10))[0] == ["x", "###x", "###yy"]
    assert _verify(solve, ("xxyy", thick, "[UNK]", "##", 10))[0] == ["[UNK]"]
    # 「词条本身就等于前缀」：start>0 时永远命中不了，start==0 时是普通词条
    odd = ["[UNK]", "##", "a", "##a"]
    assert _verify(solve, ("##", odd, "[UNK]", "##", 5))[0] == ["##"]
    assert _verify(solve, ("a#", odd, "[UNK]", "##", 5))[0] == ["[UNK]"]
    assert _verify(solve, ("##a a##", odd, "[UNK]", "##", 5))[0] == ["##a", "[UNK]"]
    # 词本身以 ## 开头时，起始位匹配的是整串，不会剥掉前缀
    assert _verify(solve, ("##a", odd, "[UNK]", "##", 5))[2] == [[0, 3]]


def check_length_limit_boundary(solve):
    vocab = ["[UNK]", "abcd", "ab", "##cd"]
    # 恰好等于上限：正常切
    assert _verify(solve, ("abcd", vocab, "[UNK]", "##", 4))[0] == ["abcd"]
    assert _verify(solve, ("abcd", vocab, "[UNK]", "##", 5))[0] == ["abcd"]
    # 超一个码点：整词 UNK，且不做任何切分尝试
    tokens, ids, offsets, word_ids = _verify(solve, ("abcde", vocab, "[UNK]", "##", 4))
    assert tokens == ["[UNK]"] and offsets == [[0, 5]] and word_ids == [0]
    # 上限为 1：长度 1 的词正常，长词整词 UNK
    tokens, ids, offsets, word_ids = _verify(solve, ("a a ab", vocab, "[UNK]", "##", 1))
    assert tokens == ["[UNK]", "[UNK]", "[UNK]"], tokens
    tight = ["[UNK]", "a", "##a", "##b", "b"]
    assert _verify(solve, ("ab", tight, "[UNK]", "##", 1))[0] == ["[UNK]"]
    assert _verify(solve, ("a b a", tight, "[UNK]", "##", 1))[0] == ["a", "b", "a"]


def check_whitespace_and_numbering(solve):
    for text, expected_words in [("", []), (" ", []), ("\n\t\r\f\v", []), ("   \u00a0 ", []),
                                 ("  the  ", ["the"]), ("the\n\tcat", ["the", "cat"]),
                                 ("the cat the", ["the", "cat", "the"])]:
        tokens, ids, offsets, word_ids = _verify(solve, (text, EX_VOCAB, "[UNK]", "##", 10))
        assert text.split() == expected_words, (text, text.split())
        assert len(word_ids) == len(tokens)
        assert max(word_ids, default=-1) == len(expected_words) - 1, (text, word_ids)
        if not expected_words:
            assert tokens == [] and ids == [] and offsets == [] and word_ids == []
    # 词号按 split 顺序连续递增，不因 UNK 跳过、也不因多片段重复
    tokens, ids, offsets, word_ids = _verify(
        solve, ("hashcat the zzz cat", EX_VOCAB, "[UNK]", "##", 10))
    assert tokens == ["[UNK]", "the", "[UNK]", "cat"], tokens
    assert word_ids == [0, 1, 2, 3], word_ids
    assert ids == [0, 1, 0, 11], ids
    assert offsets == [[0, 7], [8, 11], [12, 15], [16, 19]], offsets


def check_invalid(solve):
    def expect_raises(args, note):
        try:
            solve(*args)
        except ValueError:
            return
        raise AssertionError(f"{note} should raise ValueError")

    vocab = list(EX_VOCAB)
    expect_raises((None, vocab, "[UNK]", "##", 5), "text is None")
    expect_raises((b"abc", vocab, "[UNK]", "##", 5), "text is bytes")
    expect_raises((123, vocab, "[UNK]", "##", 5), "text is int")
    expect_raises((EX_TEXT, tuple(vocab), "[UNK]", "##", 5), "vocab is a tuple")
    expect_raises((EX_TEXT, "abc", "[UNK]", "##", 5), "vocab is a str")
    expect_raises((EX_TEXT, [], "[UNK]", "##", 5), "vocab empty")
    expect_raises((EX_TEXT, ["[UNK]", 5], "[UNK]", "##", 5), "vocab entry not str")
    expect_raises((EX_TEXT, ["[UNK]", None], "[UNK]", "##", 5), "vocab entry None")
    expect_raises((EX_TEXT, ["[UNK]", ""], "[UNK]", "##", 5), "vocab entry empty")
    expect_raises((EX_TEXT, ["[UNK]", "a", "a"], "[UNK]", "##", 5), "vocab duplicates")
    expect_raises((EX_TEXT, vocab, "nope", "##", 5), "unk not in vocab")
    expect_raises((EX_TEXT, vocab, None, "##", 5), "unk is None")
    expect_raises((EX_TEXT, vocab, 0, "##", 5), "unk is int")
    expect_raises((EX_TEXT, vocab, "[UNK]", None, 5), "prefix is None")
    expect_raises((EX_TEXT, vocab, "[UNK]", 0, 5), "prefix is int")
    expect_raises((EX_TEXT, vocab, "[UNK]", ["#"], 5), "prefix is a list")
    expect_raises((EX_TEXT, vocab, "[UNK]", "##", True), "max chars is bool")
    expect_raises((EX_TEXT, vocab, "[UNK]", "##", 1.0), "max chars is float")
    expect_raises((EX_TEXT, vocab, "[UNK]", "##", "5"), "max chars is str")
    expect_raises((EX_TEXT, vocab, "[UNK]", "##", None), "max chars is None")
    expect_raises((EX_TEXT, vocab, "[UNK]", "##", 0), "max chars zero")
    expect_raises((EX_TEXT, vocab, "[UNK]", "##", -1), "max chars negative")
    # 合法边界必须放行
    _verify(solve, ("", ["[UNK]"], "[UNK]", "", 1))
    _verify(solve, ("a", ["[UNK]", "a"], "[UNK]", "##", 1))
    _verify(solve, ("a", ["[UNK]"], "[UNK]", "##", 1))
    _verify(solve, ("x" * 3, ["[UNK]", "x", "##x"], "[UNK]", "##", 3))


def check_maximum_shape(solve):
    big = ["[UNK]"]
    for i in range(19999):
        big.append(chr(97 + i % 26) + str(i))
    words = [big[i] for i in range(1, len(big), 40)]
    text = " ".join(words)
    assert len(text) <= 4000, len(text)
    _verify(solve, (text, big, "[UNK]", "##", 12))
    # 一个 3999 码点的词，上限放行但词表里没有可命中的串 -> 整词 UNK
    _verify(solve, ("z" * 3999, big, "[UNK]", "##", 4000))
    # 上限为 1 时逐字：只有 t0 与 a 开头的单字不存在，全部走 UNK 分支
    _verify(solve, (" ".join(words[:60]), big, "[UNK]", "##", 1))
