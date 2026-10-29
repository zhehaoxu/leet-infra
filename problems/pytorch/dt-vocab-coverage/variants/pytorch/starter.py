def solve(documents, vocab, ignored_tokens):
    """中文契约：
    documents 是 0..30 篇文档组成的 list[list[str]]，每篇含 0..100 个 token。
    token 是长 1..30 的非空 Unicode str。vocab 是含 0..100 个唯一 token 的
    list[str]；只要 vocab 有重复项就抛 ValueError，即使该项被忽略或语料为空。
    ignored_tokens 是允许重复的 token 列表，其余输入保证合法。
    先移除与 ignored_tokens 中任一 token 精确相等的出现；不做小写转换、
    Unicode 归一化或子串匹配。文档（包括空文档）及其顺序均保留。

    返回恰有以下 8 个键的 Python dict：
    - vocab_counts: list[int]，保留语料中每个词表词的总出现次数。
    - vocab_doc_freq: list[int]，包含每个词表词的文档数，每篇最多贡献 1。
      这两项与 vocab 原顺序对齐，被忽略或未出现的词为 0。
    - oov_tokens: list[str]，保留语料中不属于 vocab 的唯一 token，按
      (-语料总出现次数, token 的 Python 字符串字典序) 排序；不得合并为 unknown。
    - oov_counts: list[int]，与 oov_tokens 对齐的总出现次数。
    - oov_doc_freq: list[int]，与 oov_tokens 对齐的文档频次，每篇最多贡献 1。
    - document_stats: list[tuple[int, int]]，按输入文档顺序记录
      (保留 token 出现次数, OOV token 出现次数)，空文档为 (0, 0)。
    - token_coverage: float，保留的词表内 token 出现次数 / 全部保留出现次数。
    - type_coverage: float，被观察到的保留词表内词种数 / 全部保留观察词种数。
      两种覆盖率的分母为 0 时返回 0.0。未出现的 vocab 词不计入词种分母。
    所有计数是 Python int，所有容器和数值是上述 Python 内置类型。
    不修改 documents（含内层列表）、vocab 或 ignored_tokens。

    English contract:
    documents is a list of 0..30 token lists, each containing 0..100 tokens.
    Every token is a nonempty Unicode str of length at most 30. vocab is a list
    of 0..100 unique tokens; raise ValueError on any duplicate, even if ignored
    or the corpus is empty. ignored_tokens is a token list allowing duplicates.
    All other inputs are valid. Remove occurrences exactly equal to an ignored
    token before computing statistics. Do not lowercase, normalize Unicode, or
    use substring matching. Preserve all documents, including empty ones.

    Return a Python dict with exactly these eight keys:
    - vocab_counts: list[int] of retained corpus occurrence counts per vocab
      token, aligned with the original vocab order.
    - vocab_doc_freq: list[int] in that same order, counting each document at
      most once per token. Ignored and unseen vocab tokens have zero in both.
    - oov_tokens: list[str] of distinct retained tokens absent from vocab,
      sorted by (-corpus_count, token) using Python string lexicographic order.
      Keep OOV identities separate; do not collapse them into an unknown ID.
    - oov_counts: list[int] of occurrence counts aligned with oov_tokens.
    - oov_doc_freq: list[int] of document frequencies aligned with oov_tokens.
    - document_stats: list of Python (int, int) tuples in document order:
      (retained occurrence count, OOV occurrence count); empty documents (0, 0).
    - token_coverage: float, retained in-vocab occurrences / all retained
      occurrences.
    - type_coverage: float, observed retained in-vocab types / all observed
      retained types. Unseen vocab entries are not part of this denominator.
      Each coverage is 0.0 when its denominator is zero.
    Use only the stated Python built-in types and do not mutate any input,
    including inner document lists.
    """
    raise NotImplementedError
