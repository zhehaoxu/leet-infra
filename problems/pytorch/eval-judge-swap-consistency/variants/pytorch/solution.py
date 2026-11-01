def solve(matches, models):
    """FastChat llm_judge pairwise 全链路账（解析 → swap 映射 → 结算 → 聚合）。"""

    def parse(text):
        if "[[A]]" in text:
            return "A"
        if "[[B]]" in text:
            return "B"
        if "[[C]]" in text:
            return "tie"
        return "error"

    def resolve(m):
        w1 = parse(m["g1_judgment"])
        w2 = parse(m["g2_judgment"])
        g1 = {"A": m["model_1"], "B": m["model_2"]}.get(w1, w1)
        g2 = {"A": m["model_2"], "B": m["model_1"]}.get(w2, w2)
        if g1 == "error" or g2 == "error":
            outcome = "error"
        elif g1 == "tie" or g1 != g2:
            outcome = "tie"
        else:
            outcome = g1
        return {
            "question_id": m["question_id"],
            "g1_winner": g1,
            "g2_winner": g2,
            "outcome": outcome,
        }

    rows = [resolve(m) for m in matches]

    stats = {
        name: {"win": 0, "loss": 0, "tie": 0, "win_rate_adjusted": 0.0}
        for name in models
    }
    for row in rows:
        if row["outcome"] == "error":
            continue
        m1 = next(m["model_1"] for m in matches
                  if m["question_id"] == row["question_id"])
        m2 = next(m["model_2"] for m in matches
                  if m["question_id"] == row["question_id"])
        if row["outcome"] == "tie":
            stats[m1]["tie"] += 1
            stats[m2]["tie"] += 1
        elif row["outcome"] == m1:
            stats[m1]["win"] += 1
            stats[m2]["loss"] += 1
        else:
            stats[m2]["win"] += 1
            stats[m1]["loss"] += 1

    for s in stats.values():
        total = s["win"] + s["loss"] + s["tie"]
        s["win_rate_adjusted"] = (s["win"] + 0.5 * s["tie"]) / total if total else 0.0

    return {"matches": rows, "stats": stats}
