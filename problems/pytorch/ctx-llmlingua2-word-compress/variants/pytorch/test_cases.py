import re

import numpy as np

KEYS = {"words", "word_probs", "keep_words", "keep_labels", "threshold"}
PUNCT = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')


def _pct(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _pure(token, model):
    if model == "bert-base-multilingual-cased":
        return token.lstrip("##")
    return token.lstrip("▁")


def _begin(token, model, force, map_values):
    if model == "bert-base-multilingual-cased":
        if token.lstrip("##") in force or token.lstrip("##") in map_values:
            return True
        return not token.startswith("##")
    if token in PUNCT or token in force or token in map_values:
        return True
    return token.startswith("▁")


def _oracle(tokens, token_probs, reduce_rate=0.5, token_to_word="mean",
            force_tokens=(), token_map=None, force_reserve_digit=False,
            drop_consecutive=False, special_tokens=(), model_name="xlm-roberta-large",
            token_count=None):
    token_map = {} if token_map is None else token_map
    force = list(force_tokens)
    map_values = set(token_map.values())
    words, groups = [], []
    for token, prob in zip(tokens, token_probs):
        if token in special_tokens:
            continue
        if _begin(token, model_name, force, map_values):
            pure = _pure(token, model_name)
            if pure in force or pure in map_values:
                prob = 1.0
            word = token
            for original, replacement in token_map.items():
                word = word.replace(replacement, original)
            words.append(word)
            groups.append([1.0 if force_reserve_digit and re.search(r"\d", word) else prob])
        else:
            groups[-1].append(
                1.0 if force_reserve_digit and re.search(r"\d", token) else prob)
            words[-1] += _pure(token, model_name)

    if token_to_word == "mean":
        word_probs = [sum(g) / len(g) for g in groups]
    else:
        word_probs = [g[0] for g in groups]

    if drop_consecutive:
        word_threshold = _pct(word_probs, int(100 * reduce_rate))
        between = False
        prev = None
        for i, word in enumerate(words):
            if word in force:
                if between:
                    between = False
                elif word == prev:
                    word_probs[i] = 0.0
                prev = word
            else:
                between = between or word_probs[i] > word_threshold

    expanded = []
    for word, prob in zip(words, word_probs):
        n = 1 if token_count is None else token_count(word)
        expanded += [prob] * n
    threshold = _pct(expanded, int(100 * reduce_rate + 1))

    keep_words, labels = [], []
    for word, prob in zip(words, word_probs):
        if prob > threshold or (threshold == 1.0 and prob == threshold):
            if drop_consecutive and word in force and keep_words and keep_words[-1] == word:
                labels.append(0)
            else:
                keep_words.append(word)
                labels.append(1)
        else:
            labels.append(0)
    return {
        "words": words,
        "word_probs": word_probs,
        "keep_words": keep_words,
        "keep_labels": labels,
        "threshold": threshold,
    }


def _verify(solve, *args, **kwargs):
    tokens = list(args[0])
    got = solve(*args, **kwargs)
    assert isinstance(got, dict) and set(got) == KEYS, f"键不对：{sorted(got)}"
    want = _oracle(*args, **kwargs)
    assert got["words"] == want["words"], f"words：{got['words']} vs {want['words']}"
    assert got["keep_words"] == want["keep_words"], (
        f"keep_words：{got['keep_words']} vs {want['keep_words']}")
    assert got["keep_labels"] == want["keep_labels"], (
        f"keep_labels：{got['keep_labels']} vs {want['keep_labels']}")
    assert len(got["word_probs"]) == len(want["word_probs"])
    for a, b in zip(got["word_probs"], want["word_probs"]):
        assert abs(float(a) - float(b)) < 1e-12, (a, b)
    assert abs(float(got["threshold"]) - want["threshold"]) < 1e-12, (
        got["threshold"], want["threshold"])
    assert args[0] == tokens, "不应修改输入 tokens"
    return got


def check_example(solve):
    """题面示例：xlm-roberta 四个词，reduce_rate=0.5。"""
    got = _verify(
        solve,
        ["▁Hello", "▁world", "▁Hello", "▁world"],
        [0.9, 0.4, 0.8, 0.3],
    )
    assert got["words"] == ["▁Hello", "▁world", "▁Hello", "▁world"]
    assert got["keep_words"] == ["▁Hello", "▁Hello"]
    assert got["keep_labels"] == [1, 0, 1, 0]
    assert abs(got["threshold"] - 0.612) < 1e-9, got["threshold"]


check_example.example = True


def check_matches_oracle(solve):
    """多种 model/合并模式/force/map/digit/特殊 token 组合对拍独立实现。"""
    cases = [
        dict(tokens=["▁a", "▁b", "▁c"], probs=[0.9, 0.6, 0.3]),
        dict(tokens=["▁a", "▁b", "▁c"], probs=[0.9, 0.6, 0.3], token_to_word="first"),
        dict(tokens=["hello", "##world", "##s"], probs=[0.9, 0.9, 0.2],
             model_name="bert-base-multilingual-cased"),
        dict(tokens=["hello", "##world", "##s"], probs=[0.1, 0.1, 0.1],
             model_name="bert-base-multilingual-cased", token_to_word="first",
             force_tokens=["hello"]),
        dict(tokens=[",", "▁a", "▁1", "2"], probs=[0.2, 0.7, 0.4, 0.4],
             force_reserve_digit=True),
        dict(tokens=["▁NEW", "▁tail"], probs=[0.2, 0.2],
             token_map={"▁x": "▁NEW"}, force_tokens=["NEW"]),
        dict(tokens=["<s>", "▁hello", "</s>", "▁world"], probs=[0.1, 0.9, 0.2, 0.8],
             special_tokens=["<s>", "</s>"]),
        dict(tokens=["▁a", "▁b", "▁c"], probs=[0.9, 0.6, 0.3],
             token_count={"▁a": 3, "▁b": 1, "▁c": 1}.__getitem__),
        dict(tokens=["▁a", "▁a"], probs=[1.0, 1.0], force_tokens=["▁a"],
             drop_consecutive=True),
        dict(tokens=["▁a", "▁a", "▁a", "▁b"], probs=[1.0, 1.0, 0.2, 0.9],
             force_tokens=["▁a"], drop_consecutive=True, reduce_rate=0.3),
    ]
    for case in cases:
        _verify(solve, case.pop("tokens"), case.pop("probs"), **case)


def check_force_tokens_and_map(solve):
    """force_tokens 与 token_map 命中词首时概率强制为 1.0，map 还会还原原词。"""
    got = _verify(solve, ["▁NEW", "▁tail"], [0.2, 0.2],
                  token_map={"▁x": "▁NEW"}, force_tokens=["NEW"])
    assert got["words"] == ["▁x", "▁tail"]
    assert got["word_probs"][0] == 1.0
    assert got["keep_labels"] == [1, 0]


def check_bert_continuation_merge(solve):
    """BERT 的 ## 续接子词拼到同一个词上，mean/first 两种合并结果不同。"""
    mean = _verify(solve, ["hello", "##world", "##s"], [0.9, 0.9, 0.2],
                   model_name="bert-base-multilingual-cased")
    first = _verify(solve, ["hello", "##world", "##s"], [0.9, 0.9, 0.2],
                    model_name="bert-base-multilingual-cased", token_to_word="first")
    assert mean["words"] == ["helloworlds"] and first["words"] == ["helloworlds"]
    assert abs(mean["word_probs"][0] - (0.9 + 0.9 + 0.2) / 3) < 1e-12
    assert first["word_probs"][0] == 0.9


def check_drop_consecutive(solve):
    """连续重复的 force 词只保留第一次。"""
    got = _verify(solve, ["▁a", "▁a"], [1.0, 1.0], force_tokens=["▁a"],
                  drop_consecutive=True)
    assert got["keep_labels"] == [1, 0], got
    assert got["keep_words"] == ["▁a"]


def check_special_tokens_skipped(solve):
    """特殊 token 不进入词表，也不占标签位。"""
    got = _verify(solve, ["<s>", "▁hello", "</s>", "▁world"],
                  [0.1, 0.9, 0.2, 0.8], special_tokens=["<s>", "</s>"])
    assert got["words"] == ["▁hello", "▁world"]
    assert len(got["keep_labels"]) == 2


def check_token_expansion_changes_threshold(solve):
    """token_count 把词概率展开回 token 级，展开后阈值与标签会变。"""
    tokens = ["▁a", "▁b", "▁c"]
    probs = [0.9, 0.6, 0.3]
    plain = _verify(solve, tokens, probs)
    expanded = _verify(solve, tokens, probs,
                       token_count={"▁a": 3, "▁b": 1, "▁c": 1}.__getitem__)
    assert plain["keep_labels"] == [1, 0, 0], plain
    assert expanded["keep_labels"] == [0, 0, 0], expanded
    assert abs(expanded["threshold"] - 0.9) < 1e-12, expanded["threshold"]


def check_threshold_equality_at_one(solve):
    """阈值恰为 1.0 时，等于阈值的词也要保留。"""
    got = _verify(solve, ["▁a", "▁b"], [1.0, 1.0])
    assert got["threshold"] == 1.0
    assert got["keep_labels"] == [1, 1]


def check_validation(solve):
    """长度、概率范围、rate、model/token 类型等非法输入抛 ValueError。"""
    def bad(fn):
        try:
            fn()
        except ValueError:
            return
        raise AssertionError("应抛 ValueError")

    bad(lambda: solve([], []))
    bad(lambda: solve(["▁a"], []))
    bad(lambda: solve(["▁a"], [0.5, 0.5]))
    bad(lambda: solve(["▁a"], [1.5]))
    bad(lambda: solve(["▁a"], [-0.1]))
    bad(lambda: solve(["▁a"], [True]))
    bad(lambda: solve(["▁a"], [0.5], reduce_rate=0.0))
    bad(lambda: solve(["▁a"], [0.5], reduce_rate=1.0))
    bad(lambda: solve(["▁a"], [0.5], reduce_rate=True))
    bad(lambda: solve(["▁a"], [0.5], token_to_word="median"))
    bad(lambda: solve(["▁a"], [0.5], model_name="gpt2"))
    bad(lambda: solve(["▁a"], [0.5], token_map=[("a", "b")]))
    bad(lambda: solve(["▁a"], [0.5], token_map={"a": 1}))
    bad(lambda: solve(["▁a"], [0.5], token_count=3))
    bad(lambda: solve(["▁a"], [0.5], token_count=lambda w: 0))
    bad(lambda: solve(["##a"], [0.5], model_name="bert-base-multilingual-cased"))
    bad(lambda: solve(["<s>"], [0.5], special_tokens=["<s>"]))
    bad(lambda: solve(["▁a"], [0.5], special_tokens=[1]))
    bad(lambda: solve(["▁a"], [0.5], force_tokens=[1]))
    bad(lambda: solve(["▁a"], [0.5], force_reserve_digit=1))
    bad(lambda: solve(["▁a"], [0.5], drop_consecutive=1))
    bad(lambda: solve("▁a", [0.5]))
