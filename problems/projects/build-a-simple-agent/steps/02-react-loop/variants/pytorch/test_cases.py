"""L1 fixed-fixture comparison + L2 loop invariants for step 02."""

SPEC = {
    "name": "search", "strict": True,
    "params": {
        "query": {"type": "str", "required": True},
        "top_k": {"type": "int", "required": False, "default": 3},
    },
}
TOOLS = {"search": SPEC}
SCRIPT = [
    {"thought": "need info", "action": {"tool": "search", "args": {"query": "a"}}},
    {"thought": "bad call", "action": {"tool": "search", "args": {"top_k": "three"}}},
    {"thought": "done", "final": "answer-1"},
    {"thought": "never reached", "action": {"tool": "search", "args": {"query": "b"}}},
]


def _fixture_check(solve, items, expected, limit):
    got = solve(items, TOOLS, limit)
    assert got == expected, f"expected {expected}, got {got}"
    return got


def check_basic(solve):
    expected = {
        "trace": [
            {"step": 1, "type": "action", "thought": "need info", "tool": "search",
             "args": {"query": "a"}, "valid": True},
            {"step": 2, "type": "action", "thought": "bad call", "tool": "search",
             "args": {"top_k": "three"}, "valid": False},
            {"step": 3, "type": "final", "thought": "done", "final": "answer-1"},
        ],
        "steps_used": 3, "terminated_by": "final", "final": "answer-1",
    }
    _fixture_check(solve, SCRIPT, expected, 6)


def check_budget_and_script_end(solve):
    budget = _fixture_check(solve, SCRIPT, {
        "trace": [
            {"step": 1, "type": "action", "thought": "need info", "tool": "search",
             "args": {"query": "a"}, "valid": True},
            {"step": 2, "type": "action", "thought": "bad call", "tool": "search",
             "args": {"top_k": "three"}, "valid": False},
        ],
        "steps_used": 2, "terminated_by": "budget", "final": None,
    }, 2)
    assert budget["steps_used"] == 2
    endless = [{"thought": "x", "action": {"tool": "search", "args": {"query": "a"}}}]
    out = solve(endless, TOOLS, 10)
    assert out["terminated_by"] == "script_end"
    assert out["final"] is None


def check_noop_and_unknown_tool(solve):
    items = [
        {"thought": "thinking"},
        {"thought": "unknown", "action": {"tool": "missing", "args": {}}},
        {"thought": "end", "final": ""},
    ]
    out = solve(items, TOOLS, 5)
    assert out["trace"][0]["type"] == "noop"
    assert out["trace"][1]["valid"] is False
    assert out["trace"][-1]["type"] == "final"
    assert out["final"] == ""
    assert out["terminated_by"] == "final"


check_basic.example = True


def check_property_loop_bounded(solve):
    """L2: the trace never exceeds the budget, repeats identically, has <= 1 final at the end."""
    for limit in (0, 1, 3, 8):
        out = solve(SCRIPT, TOOLS, limit)
        assert out["steps_used"] == len(out["trace"]) <= limit or limit == 0
        assert out == solve(SCRIPT, TOOLS, limit)
        finals = [i for i, item in enumerate(out["trace"]) if item["type"] == "final"]
        assert len(finals) <= 1
        assert not finals or finals[0] == len(out["trace"]) - 1
