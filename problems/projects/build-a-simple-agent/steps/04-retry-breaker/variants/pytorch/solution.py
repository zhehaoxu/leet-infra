"""Reference: retry plan + circuit breaker over fixture outcomes (step 04)."""


def solve(script, tools, outcomes, policy):
    from scaffold.react_loop import run_loop

    loop = run_loop(script, tools, policy["max_steps"])
    actions = [item["tool"] for item in loop["trace"]
               if item["type"] == "action" and item["valid"]]
    seq = list(outcomes or [])
    limit = max(1, int(policy.get("max_attempts", 3)))
    base = int(policy.get("base_delay_ms", 10))
    factor = int(policy.get("factor", 2))
    budget = int(policy.get("budget_ms", 100))
    threshold = max(1, int(policy.get("breaker_threshold", 2)))
    window = max(1, int(policy.get("window", 3)))
    attempts, delays, spent, recent = [], [], 0, []
    breaker, succeeded, why = False, False, "outcomes"
    for i in range(limit):
        if i >= len(seq):
            why = "outcomes"
            break
        outcome = seq[i]
        attempts.append(outcome)
        if outcome == "ok":
            succeeded, why = True, "success"
            break
        recent = (recent + [outcome])[-window:]
        if len(recent) >= threshold:
            breaker, why = True, "breaker"
            break
        delay = base * (factor ** len(delays))
        if spent + delay > budget:
            why = "budget"
            break
        delays.append(delay)
        spent += delay
        if i == limit - 1:
            why = "attempts"
    return {
        "actions": actions,
        "terminated_by": loop["terminated_by"],
        "retry": {
            "attempts": attempts, "delays": delays, "total_delay_ms": spent,
            "breaker_open": breaker, "succeeded": succeeded, "stopped_by": why,
        },
    }
