"""Exact path-posterior sampling under the problem's explicit uniform protocol."""

import math


def _logsumexp(values):
    if not values:
        return -math.inf
    maximum = max(values)
    return maximum + math.log(math.fsum(math.exp(x - maximum) for x in values))


def solve(text, vocab, log_scores, alpha, uniforms):
    if any(token == "" for token in vocab) or len(set(vocab)) != len(vocab):
        raise ValueError("vocabulary items must be nonempty and unique")
    n = len(text)
    if n == 0:
        return ([], 0.0, 0.0, 0)

    prefix = [-math.inf] * (n + 1)
    prefix[0] = 0.0
    # Store only tokens whose preceding prefix is reachable, in vocab ID order.
    endings = [[] for _ in range(n + 1)]
    for end in range(1, n + 1):
        weights = []
        for token_id, token in enumerate(vocab):
            start = end - len(token)
            if start >= 0 and prefix[start] != -math.inf and text[start:end] == token:
                log_weight = prefix[start] + alpha * log_scores[token_id]
                endings[end].append((token_id, start, log_weight))
                weights.append(log_weight)
        prefix[end] = _logsumexp(weights)

    if prefix[n] == -math.inf:
        raise ValueError("text has no complete segmentation")

    reversed_ids = []
    end = n
    used = 0
    while end:
        u = uniforms[used]
        used += 1
        candidates = endings[end]
        chosen = candidates[-1]  # Floating-point cumulative tail fallback.
        log_u = math.log(u) if u else -math.inf
        log_cumulative = -math.inf
        for candidate in candidates:
            log_cumulative = _logsumexp([log_cumulative, candidate[2] - prefix[end]])
            if log_cumulative > log_u:
                chosen = candidate
                break
        token_id, end, _ = chosen
        reversed_ids.append(token_id)

    ids = list(reversed(reversed_ids))
    score = math.fsum(log_scores[token_id] for token_id in ids)
    return (ids, float(prefix[n]), float(alpha * score - prefix[n]), used)
