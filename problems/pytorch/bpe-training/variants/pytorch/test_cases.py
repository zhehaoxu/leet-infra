def _norm(out):
    """把返回的每条规则规范化为 2-tuple，兼容 list/tuple。"""
    return [tuple(m) for m in out]


DATA = [("low", 5), ("lowest", 2), ("newer", 6), ("wider", 3)]


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    out = _norm(solve(DATA, 4))
    assert out == [("e", "r"), ("er", "</w>"), ("l", "o"), ("lo", "w")], out


def check_eight_merges(solve):
    out = _norm(solve(DATA, 8))
    assert out == [
        ("e", "r"), ("er", "</w>"), ("l", "o"), ("lo", "w"),
        ("e", "w"), ("ew", "er</w>"), ("n", "ewer</w>"), ("low", "</w>"),
    ], out


def check_tie_break_lexicographic(solve):
    # ('a','b') 与 ('x','y') 同频 3 → 取字典序最小的 ('a','b') 先
    out = _norm(solve([("ab", 3), ("xy", 3)], 2))
    assert out == [("a", "b"), ("ab", "</w>")], out


def check_num_merges_zero(solve):
    assert solve(DATA, 0) == []


def check_stop_when_no_pairs(solve):
    # 单词只有一符号（不可能，至少含 </w>）：给单字符词，最多合并到无相邻对
    out = _norm(solve([("a", 1)], 10))
    assert out == [("a", "</w>")], out   # 之后 (aa</w>) 只剩一个符号，无对可并，提前停


def check_order_matters(solve):
    # 学习顺序 = 输出顺序，不可排序/去重
    out = _norm(solve(DATA, 3))
    assert out[0] == ("e", "r") and out[1] == ("er", "</w>"), out


check_example.example = True
