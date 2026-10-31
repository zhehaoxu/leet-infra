import re

import numpy as np

_MODELS = ("bert-base-multilingual-cased", "xlm-roberta-large")
_PUNCT = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')


def _check_token_list(name, value):
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} 必须是 list/tuple")
    for i, token in enumerate(value):
        if not isinstance(token, str):
            raise ValueError(f"{name}[{i}] 必须是 str")
    return list(value)


def _check_prob_list(name, value, expected):
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} 必须是 list/tuple")
    if len(value) != expected:
        raise ValueError(f"{name} 长度必须与 tokens 一致")
    out = []
    for i, prob in enumerate(value):
        if isinstance(prob, bool) or not isinstance(prob, (int, float)):
            raise ValueError(f"{name}[{i}] 必须是实数")
        prob = float(prob)
        if not np.isfinite(prob) or prob < 0.0 or prob > 1.0:
            raise ValueError(f"{name}[{i}] 必须在 [0, 1] 内")
        out.append(prob)
    return out


def _pure_token(token, model_name):
    if model_name == "bert-base-multilingual-cased":
        return token.lstrip("##")
    return token.lstrip("▁")


def _is_begin_of_new_word(token, model_name, force_tokens, map_values):
    if model_name == "bert-base-multilingual-cased":
        if token.lstrip("##") in force_tokens or token.lstrip("##") in map_values:
            return True
        return not token.startswith("##")
    if token in _PUNCT or token in force_tokens or token in map_values:
        return True
    return token.startswith("▁")


def _merge_token_to_word(tokens, token_probs, special_tokens, model_name,
                         force_tokens, token_map, force_reserve_digit):
    map_values = set(token_map.values())
    words = []
    word_probs = []
    for token, prob in zip(tokens, token_probs):
        if token in special_tokens:
            continue
        if _is_begin_of_new_word(token, model_name, force_tokens, map_values):
            pure = _pure_token(token, model_name)
            if pure in force_tokens or pure in map_values:
                prob = 1.0
            word = token
            for original, replacement in token_map.items():
                word = word.replace(replacement, original)
            words.append(word)
            word_probs.append([
                1.0 if force_reserve_digit and re.search(r"\d", word) else prob
            ])
            continue
        if not words:
            raise ValueError("第一个非特殊 token 必须是词首")
        pure = _pure_token(token, model_name)
        words[-1] += pure
        word_probs[-1].append(
            1.0 if force_reserve_digit and re.search(r"\d", token) else prob
        )
    return words, word_probs


def solve(tokens, token_probs, reduce_rate=0.5, token_to_word="mean",
          force_tokens=None, token_map=None, force_reserve_digit=False,
          drop_consecutive=False, special_tokens=None,
          model_name="xlm-roberta-large", token_count=None):
    """LLMLingua-2 的词级保留概率合并与阈值筛选（CPU 参考口径）。"""
    tokens = _check_token_list("tokens", tokens)
    if not tokens:
        raise ValueError("tokens 不能为空")
    token_probs = _check_prob_list("token_probs", token_probs, len(tokens))

    if isinstance(reduce_rate, bool) or not isinstance(reduce_rate, (int, float)):
        raise ValueError("reduce_rate 必须是实数")
    reduce_rate = float(reduce_rate)
    if not np.isfinite(reduce_rate) or reduce_rate <= 0.0 or reduce_rate >= 1.0:
        raise ValueError("reduce_rate 必须在 (0, 1) 内")
    if token_to_word not in ("mean", "first"):
        raise ValueError("token_to_word 只能是 'mean' 或 'first'")
    if not isinstance(force_reserve_digit, bool):
        raise ValueError("force_reserve_digit 必须是 bool")
    if not isinstance(drop_consecutive, bool):
        raise ValueError("drop_consecutive 必须是 bool")
    if model_name not in _MODELS:
        raise ValueError(f"model_name 只支持 {_MODELS}")
    special_tokens = _check_token_list(
        "special_tokens", [] if special_tokens is None else special_tokens)
    force_tokens = _check_token_list(
        "force_tokens", [] if force_tokens is None else force_tokens)

    if token_map is None:
        token_map = {}
    if not isinstance(token_map, dict):
        raise ValueError("token_map 必须是 dict")
    for key, value in token_map.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("token_map 的键和值都必须是 str")
    if token_count is not None and not callable(token_count):
        raise ValueError("token_count 必须是可调用对象或 None")

    words, per_word = _merge_token_to_word(
        tokens, token_probs, set(special_tokens), model_name,
        force_tokens, token_map, force_reserve_digit)
    if not words:
        raise ValueError("合并后没有任何词")

    if token_to_word == "mean":
        word_probs = [sum(values) / len(values) for values in per_word]
    else:
        word_probs = [values[0] for values in per_word]

    if drop_consecutive:
        word_threshold = float(np.percentile(word_probs, int(100 * reduce_rate)))
        is_token_between = False
        prev = None
        for i, word in enumerate(words):
            if word in force_tokens:
                if is_token_between:
                    is_token_between = False
                elif word == prev:
                    word_probs[i] = 0.0
                prev = word
            else:
                is_token_between = is_token_between or word_probs[i] > word_threshold

    expanded = []
    for word, prob in zip(words, word_probs):
        count = 1 if token_count is None else token_count(word)
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("token_count 必须返回 >= 1 的 int")
        expanded.extend([prob] * count)

    threshold = float(np.percentile(expanded, int(100 * reduce_rate + 1)))

    keep_words = []
    keep_labels = []
    for word, prob in zip(words, word_probs):
        if prob > threshold or (threshold == 1.0 and prob == threshold):
            if drop_consecutive and word in force_tokens and keep_words and keep_words[-1] == word:
                keep_labels.append(0)
            else:
                keep_words.append(word)
                keep_labels.append(1)
        else:
            keep_labels.append(0)

    return {
        "words": words,
        "word_probs": word_probs,
        "keep_words": keep_words,
        "keep_labels": keep_labels,
        "threshold": threshold,
    }
