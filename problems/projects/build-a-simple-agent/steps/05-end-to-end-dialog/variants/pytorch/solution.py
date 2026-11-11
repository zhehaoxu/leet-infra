"""Reference: scripted multi-turn end-to-end acceptance (step 05, FINAL)."""


def solve(turns, tools, policy):
    from scaffold.react_loop import run_loop
    from scaffold.memory_ledger import append, compact, total_chars
    from scaffold.retry_breaker import plan

    max_steps = int(policy.get("max_steps", 6))
    keep_last = int(policy.get("keep_last", 2))
    max_chars = int(policy.get("max_chars", 24))
    retry_policy = policy.get("retry", {}) or {}
    ledger, compactions, report, total_calls = [], 0, [], 0
    for index, turn in enumerate(turns or [], 1):
        loop = run_loop(turn.get("script", []), tools, max_steps)
        calls = [item["tool"] for item in loop["trace"]
                 if item["type"] == "action" and item["valid"]]
        invalid = sum(1 for item in loop["trace"]
                      if item["type"] == "action" and not item["valid"])
        retry = plan(turn.get("outcomes", []), retry_policy)
        total_calls += len(calls)
        ledger = append(ledger, {
            "turn": index, "role": "turn",
            "text": "%s|%s" % (",".join(calls), loop["final"] or ""),
        })
        ledger, folded = compact(ledger, keep_last, max_chars)
        compactions += folded
        report.append({
            "turn": index,
            "user": turn.get("user", ""),
            "tool_calls": calls,
            "invalid_calls": invalid,
            "breakpoint": retry["stopped_by"],
            "breaker_open": retry["breaker_open"],
            "final": loop["final"],
        })
    return {
        "turns": report,
        "final_answer": report[-1]["final"] if report else None,
        "total_tool_calls": total_calls,
        "compactions": compactions,
        "ledger_chars": total_chars(ledger),
    }
