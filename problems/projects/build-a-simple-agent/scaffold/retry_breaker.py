"""Step 04 reference: deterministic retry/backoff plus a circuit breaker.

Delays are *numbers in a plan*, never real sleeps; tool outcomes arrive as a
fixture list. Everything is a pure function of (outcomes, policy).
"""


def plan(outcomes, policy):
    """Return {"attempts", "delays", "total_delay_ms", "breaker_open",
    "succeeded", "stopped_by"}.

    stopped_by is one of "success" / "attempts" / "budget" / "breaker" /
    "outcomes". Consecutive failures inside ``window`` reaching
    ``breaker_threshold`` open the breaker and stop the plan.
    """
    outcomes = list(outcomes or [])
    max_attempts = max(1, int(policy.get("max_attempts", 3)))
    base = int(policy.get("base_delay_ms", 10))
    factor = int(policy.get("factor", 2))
    budget = int(policy.get("budget_ms", 100))
    threshold = max(1, int(policy.get("breaker_threshold", 2)))
    window = max(1, int(policy.get("window", 3)))

    attempts, delays = [], []
    spent = 0
    recent = []
    breaker_open = False
    succeeded = False
    stopped_by = "outcomes"
    for i in range(max_attempts):
        if i >= len(outcomes):
            stopped_by = "outcomes"
            break
        outcome = outcomes[i]
        attempts.append(outcome)
        if outcome == "ok":
            succeeded = True
            stopped_by = "success"
            break
        recent = (recent + [outcome])[-window:]
        if len(recent) >= threshold:
            breaker_open = True
            stopped_by = "breaker"
            break
        delay = base * (factor ** len(delays))
        if spent + delay > budget:
            stopped_by = "budget"
            break
        delays.append(delay)
        spent += delay
        if i == max_attempts - 1:
            stopped_by = "attempts"
    return {
        "attempts": attempts,
        "delays": delays,
        "total_delay_ms": spent,
        "breaker_open": breaker_open,
        "succeeded": succeeded,
        "stopped_by": stopped_by,
    }
