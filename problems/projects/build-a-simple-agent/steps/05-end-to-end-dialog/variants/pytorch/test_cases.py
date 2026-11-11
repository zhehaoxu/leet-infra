"""L1 end-to-end oracle + L2 data-flow invariants for step 05 (FINAL)."""

SPEC = {
    "name": "search", "strict": True,
    "params": {"query": {"type": "str", "required": True}},
}
TOOLS = {"search": SPEC}
TURNS = [
    {"user": "find a",
     "script": [{"thought": "x", "action": {"tool": "search", "args": {"query": "a"}}},
                {"thought": "y", "final": "A"}],
     "outcomes": ["ok"]},
    {"user": "find b",
     "script": [{"thought": "x", "action": {"tool": "search", "args": {"query": "b"}}},
                {"thought": "bad", "action": {"tool": "search", "args": {}}},
                {"thought": "y", "final": "B"}],
     "outcomes": ["error", "ok"]},
]
POLICY = {
    "max_steps": 6, "keep_last": 2, "max_chars": 24,
    "retry": {"max_attempts": 3, "base_delay_ms": 10, "factor": 2,
              "budget_ms": 100, "breaker_threshold": 2, "window": 3},
}


def _oracle(turns, tools, policy):
    """Independent replay: hardcoded stage logic, no scaffold imports."""
    ledger, folds, report, calls_total = [], 0, [], 0
    for i, turn in enumerate(turns, 1):
        calls, invalid, final = [], 0, None
        for reply in turn["script"][:policy["max_steps"]]:
            if "final" in reply:
                final = reply["final"]
                break
            action = reply.get("action")
            if not action:
                continue
            args = action.get("args", {})
            ok = action["tool"] in tools and "query" in args and isinstance(args["query"], str)
            if ok:
                calls.append(action["tool"])
            else:
                invalid += 1
        outcome_seq = turn.get("outcomes", [])
        stopped = "success" if outcome_seq and outcome_seq[0] == "ok" else (
            "success" if "ok" in outcome_seq[:2] else "outcomes")
        breaker = False
        if outcome_seq[:2] == ["error", "error"]:
            breaker, stopped = True, "breaker"
        calls_total += len(calls)
        text = "%s|%s" % (",".join(calls), final or "")
        rows = ledger + [{"turn": i, "role": "turn", "text": text}]
        if len(rows) > policy["keep_last"]:
            head, tail = rows[:-policy["keep_last"]], rows[-policy["keep_last"]:]
            body = "|".join(r["text"][:policy["max_chars"]] for r in head)
            rows = [{"turn": head[-1]["turn"], "role": "summary",
                     "text": ("[%d]%s" % (len(head), body))[:policy["max_chars"] * 4]}] + tail
            folds += 1
        ledger = rows
        report.append({"turn": i, "user": turn["user"], "tool_calls": calls,
                       "invalid_calls": invalid, "breakpoint": stopped,
                       "breaker_open": breaker, "final": final})
    return {
        "turns": report,
        "final_answer": report[-1]["final"] if report else None,
        "total_tool_calls": calls_total,
        "compactions": folds,
        "ledger_chars": sum(len(r["text"]) for r in ledger),
    }


def check_basic(solve):
    got = solve(TURNS, TOOLS, POLICY)
    assert got == _oracle(TURNS, TOOLS, POLICY), f"oracle mismatch: {got}"
    assert got["final_answer"] == "B"
    assert got["total_tool_calls"] == 2
    assert got["turns"][1]["invalid_calls"] == 1


def check_breakpoint_and_breaker(solve):
    turns = [dict(TURNS[0], outcomes=["error", "error", "ok"])]
    got = solve(turns, TOOLS, POLICY)
    assert got["turns"][0]["breakpoint"] == "breaker"
    assert got["turns"][0]["breaker_open"] is True
    assert got == _oracle(turns, TOOLS, POLICY)


def check_empty_and_single(solve):
    assert solve([], TOOLS, POLICY)["final_answer"] is None
    got = solve(TURNS[:1], TOOLS, POLICY)
    assert got["compactions"] == 0
    assert got["final_answer"] == "A"


check_basic.example = True


def check_property_flow_consistency(solve):
    """L2: the report's call counts, final answer and ledger must agree."""
    for turns in (TURNS, TURNS * 2, TURNS[:1]):
        got = solve(turns, TOOLS, POLICY)
        assert got["total_tool_calls"] == sum(len(t["tool_calls"]) for t in got["turns"])
        assert got["final_answer"] == (got["turns"][-1]["final"] if got["turns"] else None)
        assert len(got["turns"]) == len(turns)
        assert got == solve(turns, TOOLS, POLICY)
        assert got["compactions"] >= 0 and got["ledger_chars"] >= 0
