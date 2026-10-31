def solve(text, vocab, unk_token, continuation_prefix, max_input_chars_per_word):
    """中文契约 / English contract

    text 是 str（0..4000 码点）；vocab 是 list[str]（1..20000 项，每项非空、互不相同，
    下标即 token id）；unk_token 是 str 且必须在 vocab 里；continuation_prefix 是 str
    （可为空串或多字符）；max_input_chars_per_word 是 python int（bool 不算）且 >= 1。
    非法输入抛 ValueError，不修改任何输入。
    text 是 str of 0..4000 code points; vocab is list[str] of 1..20000 non-empty distinct
    entries whose index is the token id; unk_token is a str present in vocab;
    continuation_prefix is a str (possibly empty or multi-char);
    max_input_chars_per_word is a python int (not bool) >= 1. Raise ValueError on invalid
    input and never mutate inputs.

    预分词：按 Python 空白（str.isspace() 的极大连续段）切词，首尾空白不产生空词，词号从 0 递增。
    Pre-tokenization: split on maximal runs of str.isspace() characters; no empty word from
    leading/trailing whitespace; word numbers increase from 0.

    逐词规则（码点数 L，词在 text 中的全局半开区间 [a, b)）：
    1. L > max_input_chars_per_word 时该词只产出一个 unk_token，偏移 [a, b)。
    2. 否则从 start=0 起，在每个 start 上取**最长**命中：候选是 word[start:end]，
       start > 0 时前面必须拼上 continuation_prefix 再查词表；命中后原样输出（含前缀），
       start 移到该命中的 end。
    3. 某个 start 上无任何命中时，**丢弃该词已产出的全部片段**，只留一个 unk_token（整词 UNK，
       不是逐字 UNK）。词表里的光杆 entry 永远不能用在 start > 0 的位置。
    Greedy longest match; on the first position with no hit the whole word becomes one unk
    token and the pieces already cut from it are dropped.

    返回 (tokens, ids, offsets, word_ids)，四个 list 等长（可为 0）：
    tokens 是 list[str]（续接片段带前缀）；ids 是 list[int]（vocab 下标）；
    offsets 是 list[[start, end]]，**码点**下标，不是 UTF-8 字节也不是 UTF-16 单元；
    word_ids 是 list[int]，片段所属词的序号。
    """
    raise NotImplementedError
