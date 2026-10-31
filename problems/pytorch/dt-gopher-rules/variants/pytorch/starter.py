PUNCTUATION_SET = frozenset(
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    "…—–·、，。！？；：""''《》【】（）〈〉「」『』"
)

STOP_WORDS = frozenset(
    ["the", "be", "to", "of", "and", "that", "have", "with"]
)


def solve(
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
    """Gopher 规则质量过滤的逐条账（口径见 starter 注释）。"""
    raise NotImplementedError
