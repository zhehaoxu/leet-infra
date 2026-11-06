"""Reference: memory ledger built on the frozen loop (step 03).

The ledger itself is THIS step's missing block, so it is implemented here; only
earlier milestones are imported from the frozen scaffold.
"""


def append(entries, record):
    return [dict(e) for e in (entries or [])] + [dict(record or {})]


def compact(entries, keep_last, max_chars):
    items = [dict(e) for e in (entries or [])]
    keep = max(0, int(keep_last))
    if len(items) <= keep:
        return items, 0
    head = items[:-keep] if keep else items
    tail = items[-keep:] if keep else []
    limit = max(1, int(max_chars))
    pieces = [str(e.get("text", ""))[:limit] for e in head]
    summary = {
        "turn": head[-1].get("turn", 0),
        "role": "summary",
        "text": ("[%d]%s" % (len(head), "|".join(pieces)))[: limit * 4],
    }
    return [summary] + tail, 1


def total_chars(entries):
    return sum(len(str(e.get("text", ""))) for e in (entries or []))


def solve(script, tools, policy):
    from scaffold.react_loop import run_loop

    loop = run_loop(script, tools, policy["max_steps"])
    ledger = []
    compactions = 0
    for item in loop["trace"]:
        if item["type"] == "final":
            text = "final:%s" % item["final"]
        elif item["type"] == "action":
            text = "action:%s" % item["tool"]
        else:
            text = "noop:"
        ledger = append(ledger, {"turn": item["step"], "role": item["type"], "text": text})
        ledger, folded = compact(ledger, policy["keep_last"], policy["max_chars"])
        compactions += folded
    return {
        "terminated_by": loop["terminated_by"],
        "final": loop["final"],
        "ledger": ledger,
        "compactions": compactions,
        "ledger_chars": total_chars(ledger),
    }
