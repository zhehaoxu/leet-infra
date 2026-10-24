MERGES = [
    ("e", "r"), ("er", "</w>"), ("l", "o"), ("lo", "w"),
    ("e", "w"), ("ew", "er</w>"), ("n", "ewer</w>"), ("low", "</w>"),
]


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    assert solve("newer", MERGES) == ["newer</w>"]
    assert solve("lowest", MERGES) == ["low", "e", "s", "t", "</w>"]


def check_wider_and_low(solve):
    assert solve("wider", MERGES) == ["w", "i", "d", "er</w>"], solve("wider", MERGES)
    assert solve("low", MERGES) == ["low</w>"], solve("low", MERGES)


def check_empty_merges_is_char_level(solve):
    assert solve("hi", []) == ["h", "i", "</w>"]


def check_unseen_word(solve):
    # 不在训练语料的词照样编码（BPE 无 OOV）
    assert solve("abc", MERGES) == ["a", "b", "c", "</w>"]


def check_partial_merge_keeps_rest(solve):
    # "lower": l o w e r </w> → (e,r)→er, (er,</w>)→er</w>, (l,o)→lo, (lo,w)→low
    assert solve("lower", MERGES) == ["low", "er</w>"], solve("lower", MERGES)


def check_idempotent_on_single(solve):
    # 单字符词：只有 (a,</w>) 若在某 merges 里才会合并；这里用不含它的规则集→原样
    assert solve("z", MERGES) == ["z", "</w>"]


check_example.example = True
