import re


def _normalize(s):
    s = s.strip()
    s = s.lower()
    s = re.sub(r'(?<=\d),(?=\d)', '', s)
    s = re.sub(r'\.$', '', s)
    s = re.sub(r'(?<!\d)(\d+)\s*(?:/|÷)\s*(?<!\d)(\d+)(?!\d)', r'\1/\2', s)
    s = re.sub(r'%$', '', s)
    s = re.sub(r'^\$', '', s)
    return s


def solve(samples, weights=None):
    """self-consistency 投票账（见 starter 注释的 8 条归一化规则）。"""
    if weights is None:
        weights = [1.0] * len(samples)
    if len(weights) != len(samples):
        raise ValueError("weights must match samples in length")
    for w in weights:
        if w < 0:
            raise ValueError("weights must be non-negative")

    normalized = [_normalize(s) for s in samples]

    tally = {}
    for ans, w in zip(normalized, weights):
        tally[ans] = tally.get(ans, 0.0) + w

    winner = None
    best = -1.0
    for ans, w in tally.items():  # dict 保序：首次出现顺序
        if w > best:
            best = w
            winner = ans

    total = sum(weights)
    share = (best / total) if total > 0 else 0.0

    return {
        "normalized": normalized,
        "tally": tally,
        "winner": winner,
        "vote_share": share,
        "n_candidates": len(tally),
    }
