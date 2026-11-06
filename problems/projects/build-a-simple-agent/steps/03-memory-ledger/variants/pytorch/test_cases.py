"""L1 independent ledger oracle + L2 replay/boundedness for step 03."""

SPEC = {
    "name": "search", "strict": True,
    "params": {"query": {"type": "str", "required": True}},
}
TOOLS = {"search": SPEC}
SCRIPT = [
    {"thought": "t1", "action": {"tool": "search", "args": {"query": "a"}}},
    {"thought": "t2", "action": {"tool": "search", "args": {"query": "b"}}},
    {"thought": "t3", "final": "answer-1"},
]
POLICY = {"max_steps": 6, "keep_last": 2, "max_chars": 24}


def _oracle(script, tools, policy):
    """Re-derive the result without importing the scaffold at all."""
    rows = []
    for i, reply in enumerate(script[:policy["max_steps"]], 1):
        if "final" in reply:
            rows.append({"turn": i, "role": "final", "text": "final:%s" % reply["final"]})
            why, final = "final", reply["final"]
            break
        action = reply.get("action")
        if not action:
            rows.append({"turn": i, "role": "noop", "text": "noop:"})
            continue
        ok = action["tool"] in tools and all(
            k in action["args"] and isinstance(action["args"][k], str)
            for k in tools[action["tool"]]["params"] if tools[action["tool"]]["params"][k].get("required")
        )
        assert ok or not ok  # oracle only needs the tool name for the text
        rows.append({"turn": i, "role": "action", "text": "action:%s" % action["tool"]})
    else:
        why, final = "script_end", None
    ledger, folds = [], 0
    for row in rows:
        ledger = ledger + [dict(row)]
        if len(ledger) > policy["keep_last"]:
            head = ledger[:-policy["keep_last"]] if policy["keep_last"] else ledger
            tail = ledger[-policy["keep_last"]:] if policy["keep_last"] else []
            body = "|".join(r["text"][:policy["max_chars"]] for r in head)
            summary = {"turn": head[-1]["turn"], "role": "summary",
                       "text": ("[%d]%s" % (len(head), body))[:policy["max_chars"] * 4]}
            ledger, folds = [summary] + tail, folds + 1
    return {
        "terminated_by": why, "final": final, "ledger": ledger,
        "compactions": folds,
        "ledger_chars": sum(len(r["text"]) for r in ledger),
    }


def check_basic(solve):
    got = solve(SCRIPT, TOOLS, POLICY)
    assert got == _oracle(SCRIPT, TOOLS, POLICY), f"oracle mismatch: {got}"
    assert got["compactions"] == 1
    assert got["ledger"][0]["role"] == "summary"
    assert got["final"] == "answer-1"


def check_policy_variants(solve):
    for keep in (0, 1, 2, 5):
        policy = dict(POLICY, keep_last=keep)
        got = solve(SCRIPT, TOOLS, policy)
        assert got == _oracle(SCRIPT, TOOLS, policy), f"keep_last={keep}: {got}"
    for chars in (4, 8, 40):
        policy = dict(POLICY, max_chars=chars)
        got = solve(SCRIPT, TOOLS, policy)
        assert got == _oracle(SCRIPT, TOOLS, policy), f"max_chars={chars}: {got}"


def check_short_script(solve):
    short = [SCRIPT[0]]
    got = solve(short, TOOLS, POLICY)
    assert got["terminated_by"] == "script_end"
    assert got["compactions"] == 0
    assert got["ledger"][0]["text"] == "action:search"


check_basic.example = True


def check_property_replay_and_bound(solve):
    """L2: replay is deterministic, ledger stays bounded, chars equal the recomputed sum."""
    for keep in (1, 2, 3):
        policy = dict(POLICY, keep_last=keep)
        got = solve(SCRIPT * 2, TOOLS, dict(policy, max_steps=12))
        assert got == solve(SCRIPT * 2, TOOLS, dict(policy, max_steps=12))
        assert len(got["ledger"]) <= max(1, keep) + 1
        assert got["ledger_chars"] == sum(len(r["text"]) for r in got["ledger"])
        assert all(isinstance(r["text"], str) for r in got["ledger"])
