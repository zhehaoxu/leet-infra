"""Step 05 reference: the end-to-end scripted-dialogue acceptance run.

This is the frozen assembly of steps 01-04. It is what the final milestone
would produce, and it doubles as the official stage vocabulary used by the L4
review step. No model calls, no wall-clock reads, no global counters.
"""

DATA_FLOW_STAGES = ("validate", "loop", "ledger", "retry", "dialog", "review")
FORBIDDEN_DECISIONS = ("replay_user_code", "judge_convergence", "call_llm")


def stages():
    return list(DATA_FLOW_STAGES)


def run_dialog(turns, tools, policy):
    """Replay fixture turns and report per-turn tool calls, breakpoints, answers."""
    from scaffold.react_loop import run_loop
    from scaffold.memory_ledger import append, compact, total_chars
    from scaffold.retry_breaker import plan

    max_steps = int(policy.get("max_steps", 6))
    keep_last = int(policy.get("keep_last", 2))
    max_chars = int(policy.get("max_chars", 24))
    retry_policy = policy.get("retry", {}) or {}
    ledger = []
    compactions = 0
    report = []
    total_calls = 0
    for turn_index, turn in enumerate(turns or [], 1):
        loop = run_loop(turn.get("script", []), tools, max_steps)
        calls = [
            item["tool"]
            for item in loop["trace"]
            if item["type"] == "action" and item["valid"]
        ]
        retry = plan(turn.get("outcomes", []), retry_policy)
        total_calls += len(calls)
        ledger = append(
            ledger,
            {
                "turn": turn_index,
                "role": "turn",
                "text": "%s|%s" % (",".join(calls), loop["final"] or ""),
            },
        )
        ledger, folded = compact(ledger, keep_last, max_chars)
        compactions += folded
        report.append(
            {
                "turn": turn_index,
                "user": turn.get("user", ""),
                "tool_calls": calls,
                "invalid_calls": sum(
                    1 for item in loop["trace"]
                    if item["type"] == "action" and not item["valid"]
                ),
                "breakpoint": retry["stopped_by"],
                "breaker_open": retry["breaker_open"],
                "final": loop["final"],
            }
        )
    return {
        "turns": report,
        "final_answer": report[-1]["final"] if report else None,
        "total_tool_calls": total_calls,
        "compactions": compactions,
        "ledger_chars": total_chars(ledger),
    }


def audit_stage_order(order):
    """Canonical order + no forbidden decisions -> accepted (L4 oracle vocabulary)."""
    order = list(order or [])
    if any(item in FORBIDDEN_DECISIONS for item in order):
        return False, "forbidden"
    return order == list(DATA_FLOW_STAGES), "order"
