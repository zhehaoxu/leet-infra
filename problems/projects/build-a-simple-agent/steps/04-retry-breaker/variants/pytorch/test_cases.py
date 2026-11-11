"""L1 independent retry oracle + L2 budget invariants for step 04."""

SPEC = {
    "name": "search", "strict": True,
    "params": {"query": {"type": "str", "required": True}},
}
TOOLS = {"search": SPEC}
SCRIPT = [
    {"thought": "a", "action": {"tool": "search", "args": {"query": "a"}}},
    {"thought": "b", "final": "done"},
]
BASE = {"max_steps": 6, "max_attempts": 4, "base_delay_ms": 10, "factor": 2,
        "budget_ms": 100, "breaker_threshold": 2, "window": 3}


def _oracle(outcomes, policy):
    seen, waits, cost, fails = [], [], 0, []
    for i in range(max(1, policy["max_attempts"])):
        if i >= len(outcomes):
            return seen, waits, cost, False, False, "outcomes"
        value = outcomes[i]
        seen.append(value)
        if value == "ok":
            return seen, waits, cost, False, True, "success"
        fails.append(value)
        if len(fails) >= max(1, policy["breaker_threshold"]):
            return seen, waits, cost, True, False, "breaker"
        wait = policy["base_delay_ms"] * (policy["factor"] ** len(waits))
        if cost + wait > policy["budget_ms"]:
            return seen, waits, cost, False, False, "budget"
        waits.append(wait)
        cost += wait
    return seen, waits, cost, False, False, "attempts"


def _compare(solve, outcomes, policy):
    got = solve(SCRIPT, TOOLS, outcomes, policy)
    seen, waits, cost, breaker, succeeded, why = _oracle(outcomes, policy)
    assert got["retry"] == {
        "attempts": seen, "delays": waits, "total_delay_ms": cost,
        "breaker_open": breaker, "succeeded": succeeded, "stopped_by": why,
    }, f"retry mismatch: {got['retry']}"
    assert got["actions"] == ["search"]
    return got


def check_basic(solve):
    _compare(solve, ["error", "error", "ok"], BASE)


def check_success_and_breaker(solve):
    got = _compare(solve, ["ok"], BASE)
    assert got["retry"]["succeeded"] is True
    got = _compare(solve, ["error", "error", "ok"], BASE)
    assert got["retry"]["breaker_open"] is True
    assert got["retry"]["stopped_by"] == "breaker"


def check_budget_and_attempts(solve):
    got = _compare(solve, ["error", "error", "error"], dict(BASE, base_delay_ms=60,
                                                           breaker_threshold=5, budget_ms=100))
    assert got["retry"]["stopped_by"] == "budget"
    got = _compare(solve, ["error", "error", "error", "error"],
                   dict(BASE, breaker_threshold=9, max_attempts=2))
    assert got["retry"]["stopped_by"] == "attempts"


check_basic.example = True


def check_property_budget_never_exceeded(solve):
    """L2: delays are a plan, never a real wait; the sum is always within budget."""
    for outcomes in (["ok"], ["error"], ["error", "ok"], ["error"] * 6):
        for base in (1, 10, 60):
            policy = dict(BASE, base_delay_ms=base, breaker_threshold=9, budget_ms=100)
            retry = solve(SCRIPT, TOOLS, outcomes, policy)["retry"]
            assert retry["total_delay_ms"] <= 100
            assert retry["total_delay_ms"] == sum(retry["delays"])
            assert retry["succeeded"] == ("ok" in retry["attempts"][:1] or
                                          retry["stopped_by"] == "success")
            assert solve(SCRIPT, TOOLS, outcomes, policy)["retry"] == retry
