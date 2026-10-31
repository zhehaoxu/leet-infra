import math


def _agg_one(steps, rule):
    if rule == "last":
        return steps[-1]
    if rule == "max":
        return max(steps)
    if rule == "min":
        return min(steps)
    if rule == "product":
        p = 1.0
        for s in steps:
            p *= s
        return p
    raise ValueError(rule)


def solve(step_scores, answer_scores=None, rule="product", top_m=1):
    """PRM best-of-N 聚合账（见 starter 注释）。"""
    if rule not in ("last", "max", "min", "product"):
        raise ValueError(f"unknown rule: {rule}")
    if not isinstance(top_m, int) or isinstance(top_m, bool) or top_m < 1:
        raise ValueError("top_m must be a positive integer")
    if not step_scores:
        raise ValueError("step_scores must be non-empty")
    if any(not s for s in step_scores):
        raise ValueError("each candidate needs at least one step")
    if answer_scores is not None and len(answer_scores) != len(step_scores):
        raise ValueError("answer_scores length mismatch")

    agg = [_agg_one(s, rule) for s in step_scores]
    if answer_scores is not None:
        agg = [a * ans for a, ans in zip(agg, answer_scores)]

    order = sorted(range(len(agg)), key=lambda i: (-agg[i], i))
    best = order[0]
    top = order[:top_m]
    top_m_val = agg[best]
    tie = sum(1 for a in agg if a == top_m_val) > 1

    return {
        "agg": agg,
        "best": best,
        "top": top,
        "n_steps": [len(s) for s in step_scores],
        "min_len": min(len(s) for s in step_scores),
        "tie": tie,
    }
