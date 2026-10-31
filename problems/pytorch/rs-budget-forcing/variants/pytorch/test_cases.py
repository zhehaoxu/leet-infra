
def _py(think_steps, budget, final_tokens, wait_steps=(), wait_cost=0):
    """独立纯 Python oracle：while 循环写法的同一状态机。"""
    emitted = 0
    forced = False
    step_emitted = []
    i = 0
    while i < len(think_steps):
        c = think_steps[i]
        if emitted + c > budget:
            step_emitted.append(budget - emitted)
            emitted = budget
            forced = True
            break
        step_emitted.append(c)
        emitted += c
        i += 1
    wait_emitted = []
    wait_log = []
    j = 0
    while (not forced) and j < len(wait_steps):
        if budget - emitted < wait_cost:
            wait_log.append(False)
            wait_emitted.append(0)
            break
        wait_log.append(True)
        emitted += wait_cost
        c = wait_steps[j]
        if emitted + c > budget:
            wait_emitted.append(budget - emitted)
            emitted = budget
            forced = True
            break
        wait_emitted.append(c)
        emitted += c
        j += 1
    total_final = sum(final_tokens)
    remaining = budget - emitted
    final_emitted = min(total_final, remaining)
    return {
        "step_emitted": step_emitted,
        "think_emitted": sum(step_emitted),
        "forced_final": forced,
        "waits_used": sum(wait_log),
        "wait_log": wait_log,
        "wait_emitted": wait_emitted,
        "final_emitted": final_emitted,
        "answer_truncated": total_final > remaining,
        "total": emitted + final_emitted,
    }


def _compare(solve, think_steps, budget, final_tokens, wait_steps=(), wait_cost=0):
    led = solve(list(think_steps), budget, list(final_tokens),
                list(wait_steps), wait_cost)
    ref = _py(think_steps, budget, final_tokens, wait_steps, wait_cost)
    assert led == ref, f"\nled={led}\nref={ref}"
    return led


def check_forced_mid_step(solve):
    # 步 [30, 40]、预算 60：第二步被切成 30，强制 Final Answer，回答全被吞
    led = _compare(solve, [30, 40], 60, [10])
    assert led["step_emitted"] == [30, 30]
    assert led["think_emitted"] == 60
    assert led["forced_final"] is True
    assert led["final_emitted"] == 0
    assert led["answer_truncated"] is True
    assert led["total"] == 60


def check_natural_end_with_wait(solve):
    # 步 [20,20]、预算 70、Wait 代价 5 + 步 15：wait 后整步发射，回答剩 8
    led = _compare(solve, [20, 20], 70, [8], [15], 5)
    assert led["step_emitted"] == [20, 20]
    assert led["wait_log"] == [True]
    assert led["waits_used"] == 1
    assert led["wait_emitted"] == [15]
    assert led["think_emitted"] == 40  # 只含思考步，Wait 步记在 wait_emitted
    assert led["forced_final"] is False
    assert led["final_emitted"] == 8
    assert led["answer_truncated"] is False
    assert led["total"] == 68  # 40 + 5 + 15 + 8


def check_wait_rejected(solve):
    # 剩余 2 < wait_cost 3：Wait 拒绝，回答用剩余发射
    led = _compare(solve, [10], 12, [1], [1], 3)
    assert led["wait_log"] == [False]
    assert led["waits_used"] == 0
    assert led["final_emitted"] == 1
    assert led["total"] == 11


def check_wait_triggers_forced(solve):
    # Wait 后步再越界：wait 代价 4 + 步切 6，截断后回答 0
    led = _compare(solve, [30, 20], 60, [5], [10], 4)
    assert led["wait_log"] == [True]
    assert led["wait_emitted"] == [6]
    assert led["forced_final"] is True
    assert led["final_emitted"] == 0
    assert led["total"] == 60


def check_forced_skips_waits(solve):
    # 思考阶段已截断：Wait 全部跳过
    led = _compare(solve, [50, 50], 60, [5], [5, 5], 1)
    assert led["forced_final"] is True
    assert led["wait_log"] == []
    assert led["waits_used"] == 0
    assert led["wait_emitted"] == []
    assert led["step_emitted"] == [50, 10]


def check_boundary_exact_fit(solve):
    # 步恰好填满预算：<= 语义不截断
    led = _compare(solve, [10, 20], 30, [0])
    assert led["step_emitted"] == [10, 20]
    assert led["forced_final"] is False
    assert led["total"] == 30
    # Wait 恰好用光剩余：remaining == wait_cost 允许
    led2 = _compare(solve, [10], 15, [0], [5], 5)
    assert led2["wait_log"] == [True]
    assert led2["total"] == 15


def check_no_waits_and_empty_answer(solve):
    led = _compare(solve, [5, 6, 7], 100, [])
    assert led["wait_log"] == []
    assert led["final_emitted"] == 0
    assert led["answer_truncated"] is False
    assert led["total"] == 18


def check_oracle_random(solve):
    import random
    rng = random.Random(123)
    for _ in range(200):
        steps = [rng.randint(1, 30) for _ in range(rng.randint(0, 6))]
        budget = rng.randint(1, 120)
        final = [rng.randint(0, 20) for _ in range(rng.randint(0, 3))]
        waits = [rng.randint(1, 25) for _ in range(rng.randint(0, 3))]
        wc = rng.randint(0, 10)
        led = solve(list(steps), budget, list(final), list(waits), wc)
        ref = _py(steps, budget, final, waits, wc)
        assert led == ref, f"steps={steps} budget={budget} final={final} waits={waits} wc={wc}"


def check_invalid_inputs(solve):
    bad = [
        lambda: solve([0.5], 10, []),
        lambda: solve([True], 10, []),
        lambda: solve([-1], 10, []),
        lambda: solve([1], 0, []),
        lambda: solve([1], True, []),
        lambda: solve([1], 10, [2], [3], -1),
        lambda: solve([1], 10, [-2]),
        lambda: solve("12", 10, []),
        lambda: solve([1], 10, [1], wait_cost=True),
    ]
    for fn in bad:
        try:
            fn()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")
