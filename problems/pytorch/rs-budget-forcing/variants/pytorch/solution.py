
def solve(think_steps, budget, final_tokens, wait_steps=(), wait_cost=0):
    for name, seq in (("think_steps", think_steps), ("final_tokens", final_tokens),
                      ("wait_steps", wait_steps)):
        if not isinstance(seq, (list, tuple)):
            raise ValueError(f"{name} must be a list of ints")
        for x in seq:
            if isinstance(x, bool) or not isinstance(x, int) or x < 0:
                raise ValueError(f"{name} entries must be non-negative ints")
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise ValueError("budget must be a positive int")
    if isinstance(wait_cost, bool) or not isinstance(wait_cost, int) or wait_cost < 0:
        raise ValueError("wait_cost must be a non-negative int")

    emitted = 0
    forced = False
    step_emitted = []
    for c in think_steps:
        if emitted + c <= budget:
            step_emitted.append(c)
            emitted += c
        else:
            step_emitted.append(budget - emitted)
            emitted = budget
            forced = True
            break

    wait_emitted = []
    wait_log = []
    if not forced:
        for c in wait_steps:
            if budget - emitted < wait_cost:
                wait_log.append(False)
                wait_emitted.append(0)
                break
            wait_log.append(True)
            emitted += wait_cost
            if emitted + c <= budget:
                wait_emitted.append(c)
                emitted += c
            else:
                wait_emitted.append(budget - emitted)
                emitted = budget
                forced = True
                break

    total_final = sum(final_tokens)
    remaining = budget - emitted
    final_emitted = min(total_final, remaining)
    answer_truncated = total_final > remaining
    return {
        "step_emitted": step_emitted,
        "think_emitted": sum(step_emitted),
        "forced_final": forced,
        "waits_used": sum(1 for x in wait_log if x),
        "wait_log": wait_log,
        "wait_emitted": wait_emitted,
        "final_emitted": final_emitted,
        "answer_truncated": answer_truncated,
        "total": emitted + final_emitted,
    }
