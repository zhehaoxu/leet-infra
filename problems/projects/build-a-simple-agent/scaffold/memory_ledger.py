"""Step 03 reference: an append-only memory ledger with deterministic compaction.

No module-level state: compact always returns a *new* ledger. The summary text
is built from the entries themselves, never from an LLM.
"""


def append(entries, record):
    """Return a new ledger with a copy of ``record`` appended."""
    out = [dict(e) for e in (entries or [])]
    out.append(dict(record or {}))
    return out


def compact(entries, keep_last, max_chars):
    """Fold all but the newest ``keep_last`` entries into one summary entry."""
    items = [dict(e) for e in (entries or [])]
    keep = max(0, int(keep_last))
    if len(items) <= keep:
        return items, 0
    head, tail = items[:-keep] if keep else items, items[-keep:] if keep else []
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
