PUNCTUATION_SET = frozenset(
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    "…—–·、，。！？；：""''《》【】（）〈〉「」『』"
)

STOP_WORDS = frozenset(
    ["the", "be", "to", "of", "and", "that", "have", "with"]
)


def _rule_checks(text, words, non_symbol_words, **p):
    lines = text.splitlines()
    n_words = len(words)
    checks = {}
    checks["short_doc"] = len(non_symbol_words) >= p["min_doc_words"]
    checks["long_doc"] = len(non_symbol_words) <= p["max_doc_words"]
    if non_symbol_words:
        avg = sum(len(w) for w in non_symbol_words) / len(non_symbol_words)
    else:
        avg = 0.0
    checks["avg_word_length"] = (
        p["min_avg_word_length"] <= avg <= p["max_avg_word_length"]
    )
    checks["symbol_word_ratio"] = (
        text.count("#") / n_words <= p["max_symbol_word_ratio"]
        if n_words
        else True
    )
    checks["ellipsis_word_ratio"] = (
        (text.count("...") + text.count("…")) / n_words <= p["max_symbol_word_ratio"]
        if n_words
        else True
    )
    if lines:
        bullet = sum(
            1 for s in lines if s.lstrip().startswith("•") or s.lstrip().startswith("-")
        ) / len(lines)
        end_ellipsis = sum(
            1 for s in lines if s.rstrip().endswith("...") or s.rstrip().endswith("…")
        ) / len(lines)
    else:
        bullet = 0.0
        end_ellipsis = 0.0
    checks["bullet_lines"] = bullet <= p["max_bullet_lines_ratio"]
    checks["end_ellipsis_lines"] = end_ellipsis <= p["max_ellipsis_lines_ratio"]
    alpha = (
        sum(1 for w in words if any(c.isalpha() for c in w)) / n_words
        if n_words
        else 0.0
    )
    checks["alpha_words"] = alpha >= p["max_non_alpha_words_ratio"]
    checks["stop_words"] = (
        len(STOP_WORDS.intersection(words)) >= p["min_stop_words"]
    )
    return checks, avg, bullet, end_ellipsis, alpha


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
    params = dict(
        min_doc_words=min_doc_words,
        max_doc_words=max_doc_words,
        min_avg_word_length=min_avg_word_length,
        max_avg_word_length=max_avg_word_length,
        max_symbol_word_ratio=max_symbol_word_ratio,
        max_bullet_lines_ratio=max_bullet_lines_ratio,
        max_ellipsis_lines_ratio=max_ellipsis_lines_ratio,
        max_non_alpha_words_ratio=max_non_alpha_words_ratio,
        min_stop_words=min_stop_words,
    )
    words = text.split()
    non_symbol_words = [w for w in words if any(ch not in PUNCTUATION_SET for ch in w)]
    checks, avg, bullet, end_ellipsis, alpha = _rule_checks(
        text, words, non_symbol_words, **params
    )

    order = [
        "short_doc",
        "long_doc",
        "avg_word_length",
        "symbol_word_ratio",
        "ellipsis_word_ratio",
        "bullet_lines",
        "end_ellipsis_lines",
        "alpha_words",
        "stop_words",
    ]
    reasons = {
        "short_doc": "gopher_short_doc",
        "long_doc": "gopher_long_doc",
        "avg_word_length": "gopher_avg_word_length",
        "symbol_word_ratio": "gopher_too_many_hashes",
        "ellipsis_word_ratio": "gopher_too_many_ellipsis",
        "bullet_lines": "gopher_too_many_bullets",
        "end_ellipsis_lines": "gopher_too_many_end_ellipsis",
        "alpha_words": "gopher_below_alpha_threshold",
        "stop_words": "gopher_enough_stop_words",
    }
    failed = next((r for c in order if not checks[c] for r in (reasons[c],)), None)
    return {
        "passed": failed is None,
        "failed_rule": failed,
        "checks": {c: checks[c] for c in order},
        "n_words": len(words),
        "n_non_symbol_words": len(non_symbol_words),
        "avg_word_length": avg,
        "bullet_lines_ratio": bullet,
        "end_ellipsis_lines_ratio": end_ellipsis,
        "alpha_words_ratio": alpha,
        "n_stop_words": len(STOP_WORDS.intersection(words)),
    }
