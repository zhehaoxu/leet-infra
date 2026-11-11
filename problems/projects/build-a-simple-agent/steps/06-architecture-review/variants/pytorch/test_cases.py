"""L1 graded design-review questions + L2 determinism for step 06."""

STAGES = ["validate", "loop", "ledger", "retry", "dialog", "review"]


def check_basic(solve):
    got = solve(STAGES, STAGES)
    assert got == {"accepted": True, "reason": "ok", "stages": STAGES, "trace_len": 6}


def check_wrong_order(solve):
    got = solve(list(reversed(STAGES)), STAGES)
    assert got["accepted"] is False
    assert got["reason"] == "order"
    assert got["stages"] == STAGES


def check_forbidden_decision(solve):
    got = solve(STAGES + ["replay_user_code"], STAGES)
    assert got["accepted"] is False
    assert got["reason"] == "forbidden"
    got = solve(["call_llm"] + STAGES, STAGES)
    assert got["reason"] == "forbidden"


def check_trace_mismatch(solve):
    got = solve(STAGES, STAGES[:-1])
    assert got["accepted"] is False
    assert got["reason"] == "trace"
    assert got["trace_len"] == 5


check_basic.example = True


def check_property_deterministic_grading(solve):
    """L2: review questions are pure functions of (decisions, trace)."""
    cases = [
        (STAGES, STAGES),
        (STAGES[1:] + STAGES[:1], STAGES),
        (STAGES + ["judge_convergence"], []),
        ([], STAGES),
    ]
    for decisions, trace in cases:
        first = solve(decisions, trace)
        assert first == solve(decisions, trace)
        assert first["stages"] == STAGES
        assert first["accepted"] in (True, False)
        assert first["reason"] in ("ok", "order", "forbidden", "trace")
