import math
import random


def _oracle(step_scores, answer_scores, rule, top_m):
    def agg_one(steps):
        if rule == "last":
            return steps[-1]
        if rule == "max":
            return max(steps)
        if rule == "min":
            return min(steps)
        p = 1.0
        for s in steps:
            p *= s
        return p

    agg = [agg_one(s) for s in step_scores]
    if answer_scores is not None:
        agg = [a * x for a, x in zip(agg, answer_scores)]
    order = sorted(range(len(agg)), key=lambda i: (-agg[i], i))
    top_val = agg[order[0]]
    return {
        "agg": agg,
        "best": order[0],
        "top": order[:top_m],
        "n_steps": [len(s) for s in step_scores],
        "min_len": min(len(s) for s in step_scores),
        "tie": agg.count(top_val) > 1,
    }


def check_anchor_four_rules(solve):
    steps = [[0.9, 0.5, 0.8], [0.7, 0.9], [1.0, 0.1, 0.1, 0.1]]
    exp = {
        "last": [0.8, 0.9, 0.1],
        "max": [0.9, 0.9, 1.0],
        "min": [0.5, 0.7, 0.1],
        "product": [0.9 * 0.5 * 0.8, 0.7 * 0.9, 1.0 * 0.1 * 0.1 * 0.1],
    }
    for rule, vals in exp.items():
        led = solve(steps, rule=rule)
        for got, want in zip(led["agg"], vals):
            assert abs(got - want) < 1e-12, (rule, got, want)
    # 四种规则各自的最佳：
    assert solve(steps, rule="last")["best"] == 1
    assert solve(steps, rule="max")["best"] == 2
    assert solve(steps, rule="min")["best"] == 1
    assert solve(steps, rule="product")["best"] == 1  # 0.63 > 0.36
    # 顶部排序（product 降序，并列取小 index）
    assert solve(steps, rule="product", top_m=3)["top"] == [1, 0, 2]


def check_answer_score_multiply(solve):
    steps = [[0.8], [0.4]]
    led = solve(steps, answer_scores=[0.5, 2.0], rule="last")
    assert abs(led["agg"][0] - 0.4) < 1e-12
    assert abs(led["agg"][1] - 0.8) < 1e-12
    assert led["best"] == 1
    # 不给 answer_scores 时不乘
    led2 = solve(steps, rule="last")
    assert led2["agg"] == [0.8, 0.4]


def check_length_ledger(solve):
    steps = [[1.0], [0.5, 0.5], [0.2] * 7]
    led = solve(steps, rule="product")
    assert led["n_steps"] == [1, 2, 7]
    assert led["min_len"] == 1
    # 长短候选同 product 值：product 归并对步长不敏感（只看值）
    assert abs(led["agg"][1] - 0.25) < 1e-12


def check_tie_and_order(solve):
    steps = [[0.5], [0.5], [0.3], [0.5]]
    led = solve(steps, rule="last", top_m=4)
    assert led["tie"] is True
    assert led["top"] == [0, 1, 3, 2]  # 并列取小 index，0.3 殿后
    assert led["best"] == 0
    # 无并列
    led2 = solve([[0.9], [0.8]], rule="last")
    assert led2["tie"] is False


def check_random_against_oracle(solve):
    rng = random.Random(655)
    for _ in range(40):
        n = rng.randint(1, 5)
        steps = [[round(rng.uniform(0.0, 1.0), 3) for _ in range(rng.randint(1, 6))]
                 for _ in range(n)]
        rule = rng.choice(["last", "max", "min", "product"])
        top_m = rng.randint(1, n)
        has_ans = rng.random() < 0.5
        ans = [round(rng.uniform(0.1, 2.0), 3) for _ in range(n)] if has_ans else None
        led = solve(steps, answer_scores=ans, rule=rule, top_m=top_m)
        exp = _oracle(steps, ans, rule, top_m)
        assert led["best"] == exp["best"], (steps, ans, rule)
        assert led["top"] == exp["top"]
        assert led["tie"] == exp["tie"]
        assert led["n_steps"] == exp["n_steps"] and led["min_len"] == exp["min_len"]
        for g, e in zip(led["agg"], exp["agg"]):
            assert abs(g - e) < 1e-9


def check_product_penalizes_long_failure(solve):
    # product 口径下任何一步接近 0 都压垮长链——这是 PRM 论文偏好 product 的机理
    steps = [[0.99] * 9 + [0.01], [0.9] * 2]
    led = solve(steps, rule="product")
    assert led["best"] == 1
    # 而 max 口径下长链仍然赢
    led2 = solve(steps, rule="max")
    assert led2["best"] == 0


def check_invalid_inputs(solve):
    steps = [[0.5], [0.6]]
    for kwargs in (
        {"rule": "mean"},
        {"top_m": 0},
        {"top_m": -1},
    ):
        try:
            solve(steps, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {kwargs}")
    try:
        solve([], rule="last")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for empty step_scores")
    try:
        solve([[0.5], []], rule="last")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for empty candidate")
    try:
        solve(steps, answer_scores=[1.0], rule="last")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for answer_scores mismatch")
