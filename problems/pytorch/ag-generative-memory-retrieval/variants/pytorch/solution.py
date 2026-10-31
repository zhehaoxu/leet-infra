import math


def solve(query_emb, memories, k, recency_decay=0.995, gw=(0.5, 3, 2)):
    if not isinstance(query_emb, (list, tuple)) or len(query_emb) == 0:
        raise ValueError("query_emb must be a non-empty list")
    if any(not isinstance(x, (int, float)) or isinstance(x, bool) for x in query_emb):
        raise ValueError("query_emb entries must be numbers")
    if not isinstance(memories, (list, tuple)) or len(memories) == 0:
        raise ValueError("memories must be a non-empty list")
    D = len(query_emb)
    for m in memories:
        if not isinstance(m, dict) or set(m.keys()) != {"emb", "importance"}:
            raise ValueError("memories entries need keys emb/importance")
        if not isinstance(m["emb"], (list, tuple)) or len(m["emb"]) != D:
            raise ValueError("emb dim mismatch")
        if any(not isinstance(x, (int, float)) or isinstance(x, bool) for x in m["emb"]):
            raise ValueError("emb entries must be numbers")
        if isinstance(m["importance"], bool) or not isinstance(m["importance"], (int, float)):
            raise ValueError("importance must be a number")
        if math.isnan(float(m["importance"])) or math.isinf(float(m["importance"])):
            raise ValueError("importance must be finite")
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive int")
    if (isinstance(recency_decay, bool) or not isinstance(recency_decay, (int, float))
            or not (0 < float(recency_decay) <= 1)):
        raise ValueError("recency_decay must be in (0, 1]")
    if not isinstance(gw, (list, tuple)) or len(gw) != 3:
        raise ValueError("gw must have 3 weights")

    n = len(memories)
    recency = [float(recency_decay) ** (i + 1) for i in range(n)]
    relevance = []
    for m in memories:
        e = [float(x) for x in m["emb"]]
        q = [float(x) for x in query_emb]
        dot = sum(a * b for a, b in zip(e, q))
        ne = math.sqrt(sum(a * a for a in e))
        nq = math.sqrt(sum(b * b for b in q))
        relevance.append(dot / (ne * nq))
    importance = [float(m["importance"]) for m in memories]

    def norm01(vals):
        lo = min(vals)
        hi = max(vals)
        rng = hi - lo
        if rng == 0:
            return [0.5] * len(vals)
        return [(v - lo) * 1.0 / rng for v in vals]

    rec_n = norm01(recency)
    rel_n = norm01(relevance)
    imp_n = norm01(importance)

    scores = [gw[0] * rec_n[i] + gw[1] * rel_n[i] + gw[2] * imp_n[i] for i in range(n)]
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)
    top = order[:k]
    return {
        "scores": scores,
        "top": top,
        "components": {"recency": rec_n, "relevance": rel_n, "importance": imp_n},
    }
