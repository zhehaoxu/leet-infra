"""参考解：手工扫描空白切词，逐词贪心最长命中，失败即整词 UNK。

四个口径点：

1. **最长命中不是最少片段**。每个位置只看「以该位置开头的最长词表串」，不回溯、不比较总片段数，
   所以存在某种切分并不代表贪心能切成。
2. **续接位必须拼前缀再查表**：词表里的光杆 `cat` 只能出现在词的起始位置。
3. **失败是词级的**：某个位置无命中时，该词已经切出的片段全部作废，只留一个覆盖整词的
   `unk_token`；输出里不会出现「若干片段 + `[UNK]`」的混合。
4. **长度上限按码点算**，Python 的 `len(str)` 已经是码点数，不需要任何编码转换。

`max_span` 只是将候选端点压到「词表最长串」以内（起始位）或再减去前缀长度（续接位），
它不改变任何一次命中结果：更长的候选必然不在词表里。写错这一上界会让长词漏掉命中，
属于典型的过度剪枝。
"""


def _check(text, vocab, unk_token, continuation_prefix, max_input_chars_per_word):
    if not isinstance(text, str):
        raise ValueError("text must be a str")
    if not isinstance(vocab, list) or not vocab:
        raise ValueError("vocab must be a non-empty list")
    for entry in vocab:
        if not isinstance(entry, str):
            raise ValueError("vocab entries must be str")
        if not entry:
            raise ValueError("vocab entries must be non-empty")
    if len(set(vocab)) != len(vocab):
        raise ValueError("vocab must not contain duplicate entries")
    if not isinstance(unk_token, str) or unk_token not in vocab:
        raise ValueError("unk_token must be a str present in vocab")
    if not isinstance(continuation_prefix, str):
        raise ValueError("continuation_prefix must be a str")
    if isinstance(max_input_chars_per_word, bool) or not isinstance(max_input_chars_per_word, int):
        raise ValueError("max_input_chars_per_word must be a python int")
    if max_input_chars_per_word < 1:
        raise ValueError("max_input_chars_per_word must be at least 1")


def _split_words(text):
    """手工扫描：返回 [(word, begin, end), ...]，端点是码点下标"""
    words = []
    cursor = 0
    length = len(text)
    while cursor < length:
        if text[cursor].isspace():
            cursor += 1
            continue
        begin = cursor
        while cursor < length and not text[cursor].isspace():
            cursor += 1
        words.append((text[begin:cursor], begin, cursor))
    return words


def _segment(word, index, unk_token, continuation_prefix, max_chars, longest_entry):
    """返回该词的 [(piece, local_start, local_end), ...]；失败时返回单个覆盖整词的 UNK"""
    size = len(word)
    if size > max_chars:
        return [(unk_token, 0, size)]
    pieces = []
    start = 0
    while start < size:
        if start == 0:
            max_span = longest_entry
        else:
            max_span = longest_entry - len(continuation_prefix)
        hit = 0
        if max_span >= 1:
            high = min(size, start + max_span)
            for end in range(high, start, -1):
                piece = word[start:end] if start == 0 else continuation_prefix + word[start:end]
                if piece in index:
                    hit = end
                    pieces.append((piece, start, end))
                    break
        if hit == 0:
            return [(unk_token, 0, size)]
        start = hit
    return pieces


def solve(text, vocab, unk_token, continuation_prefix, max_input_chars_per_word):
    _check(text, vocab, unk_token, continuation_prefix, max_input_chars_per_word)
    index = {piece: position for position, piece in enumerate(vocab)}
    longest_entry = max(len(piece) for piece in vocab)

    tokens, ids, offsets, word_ids = [], [], [], []
    for word_no, (word, begin, end) in enumerate(_split_words(text)):
        for piece, local_start, local_end in _segment(
                word, index, unk_token, continuation_prefix,
                max_input_chars_per_word, longest_entry):
            tokens.append(piece)
            ids.append(index[piece])
            offsets.append([begin + local_start, begin + local_end])
            word_ids.append(word_no)
    return tokens, ids, offsets, word_ids
