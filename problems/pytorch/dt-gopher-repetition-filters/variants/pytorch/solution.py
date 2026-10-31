from collections import Counter

_DEFAULT_THRESHOLDS = {
    "duplicate_line_fraction": 0.30,
    "duplicate_paragraph_fraction": 0.30,
    "duplicate_line_character_fraction": 0.20,
    "duplicate_paragraph_character_fraction": 0.20,
    "top_2_gram_character_fraction": 0.20,
    "top_3_gram_character_fraction": 0.18,
    "top_4_gram_character_fraction": 0.16,
    "duplicate_5_gram_character_fraction": 0.15,
    "duplicate_6_gram_character_fraction": 0.14,
    "duplicate_7_gram_character_fraction": 0.13,
    "duplicate_8_gram_character_fraction": 0.12,
    "duplicate_9_gram_character_fraction": 0.11,
    "duplicate_10_gram_character_fraction": 0.10,
}
_METRIC_ORDER = list(_DEFAULT_THRESHOLDS.keys())


def _merge_span_length(spans):
    """重叠区间合并后的总覆盖长度。spans: [(s, e), ...]，s < e。"""
    if not spans:
        return 0
    spans = sorted(spans)
    total = 0
    cur_s, cur_e = spans[0]
    for s, e in spans[1:]:
        if s > cur_e:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            if e > cur_e:
                cur_e = e
    total += cur_e - cur_s
    return total


def _words_with_spans(text):
    """词 = 空白分隔的最大非空白连续段；返回 [(word, start, end), ...]。"""
    words = []
    i, n = 0, len(text)
    while i < n:
        if text[i].isspace():
            i += 1
            continue
        j = i
        while j < n and not text[j].isspace():
            j += 1
        words.append((text[i:j], i, j))
        i = j
    return words


def solve(text, thresholds=None):
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    if thresholds is not None:
        if not isinstance(thresholds, dict):
            raise ValueError("thresholds must be a dict")
        for k, v in thresholds.items():
            if k not in _DEFAULT_THRESHOLDS:
                raise ValueError(f"unknown metric: {k}")
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(f"threshold for {k} must be a number")
            if v < 0 or v > 1:
                raise ValueError(f"threshold for {k} must be in [0, 1]")
    thr = dict(_DEFAULT_THRESHOLDS)
    if thresholds:
        thr.update(thresholds)

    # ---- 行账 ----
    lines = text.splitlines()
    line_char_total = sum(len(l) for l in lines)
    line_cnt = Counter(lines)
    dup_line_instances = [l for l in lines if line_cnt[l] >= 2]
    dup_line_frac = (len(dup_line_instances) / len(lines)) if lines else 0.0
    dup_line_char_frac = (
        sum(len(l) for l in dup_line_instances) / line_char_total
        if line_char_total else 0.0
    )

    # ---- 段账：空行分隔的极大非空行块 ----
    paras = []
    cur = []
    for l in lines:
        if l.strip() == "":
            if cur:
                paras.append(cur)
                cur = []
        else:
            cur.append(l)
    if cur:
        paras.append(cur)
    para_strs = ["\n".join(block) for block in paras]
    para_chars = [sum(len(l) for l in block) for block in paras]
    para_char_total = sum(para_chars)
    para_cnt = Counter(para_strs)
    dup_para_count = sum(1 for p in para_strs if para_cnt[p] >= 2)
    dup_para_chars = sum(c for p, c in zip(para_strs, para_chars) if para_cnt[p] >= 2)
    dup_para_frac = (dup_para_count / len(para_strs)) if para_strs else 0.0
    dup_para_char_frac = dup_para_chars / para_char_total if para_char_total else 0.0

    # ---- word n-gram 账 ----
    words = _words_with_spans(text)
    total_chars = len(text)
    metrics = {
        "duplicate_line_fraction": dup_line_frac,
        "duplicate_paragraph_fraction": dup_para_frac,
        "duplicate_line_character_fraction": dup_line_char_frac,
        "duplicate_paragraph_character_fraction": dup_para_char_frac,
    }
    # top n-gram (n=2..4)：出现次数最多，并列取首现最早；覆盖 = 其全部出现实例区间并集
    for n in (2, 3, 4):
        gram_spans = {}
        gram_first = {}
        for i in range(len(words) - n + 1):
            key = tuple(w[0] for w in words[i:i + n])
            span = (words[i][1], words[i + n - 1][2])
            gram_spans.setdefault(key, []).append(span)
            if key not in gram_first:
                gram_first[key] = span[0]
        if not gram_spans:
            metrics[f"top_{n}_gram_character_fraction"] = 0.0
            continue
        best_count = max(len(v) for v in gram_spans.values())
        candidates = [k for k, v in gram_spans.items() if len(v) == best_count]
        best = min(candidates, key=lambda k: gram_first[k])
        metrics[f"top_{n}_gram_character_fraction"] = (
            _merge_span_length(gram_spans[best]) / total_chars if total_chars else 0.0
        )
    # duplicate n-gram (n=5..10)：全部 count>=2 的 n-gram 的全部出现实例合并
    for n in range(5, 11):
        all_spans = []
        gram_cnt = Counter()
        spans_by_key = {}
        for i in range(len(words) - n + 1):
            key = tuple(w[0] for w in words[i:i + n])
            gram_cnt[key] += 1
            spans_by_key.setdefault(key, []).append((words[i][1], words[i + n - 1][2]))
        for key, c in gram_cnt.items():
            if c >= 2:
                all_spans.extend(spans_by_key[key])
        metrics[f"duplicate_{n}_gram_character_fraction"] = (
            _merge_span_length(all_spans) / total_chars if total_chars else 0.0
        )

    reject_reason = None
    for name in _METRIC_ORDER:
        if metrics[name] > thr[name]:
            reject_reason = name
            break
    return {
        "accepted": reject_reason is None,
        "reject_reason": reject_reason,
        "metrics": metrics,
    }
