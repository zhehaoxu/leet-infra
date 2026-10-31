import hashlib
import json


def _canon(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, float):
        return round(v, 6)
    if isinstance(v, dict):
        return {k: _canon(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_canon(x) for x in v]
    return v


def solve(calls, threshold):
    if not isinstance(calls, (list, tuple)) or len(calls) == 0:
        raise ValueError("calls must be a non-empty list")
    if isinstance(threshold, bool) or not isinstance(threshold, int) or threshold < 0:
        raise ValueError("threshold must be a non-negative int")
    for c in calls:
        if not isinstance(c, dict) or set(c.keys()) != {"tool", "args"}:
            raise ValueError("calls entries need exactly keys tool/args")
        if not isinstance(c["tool"], str) or not c["tool"]:
            raise ValueError("tool must be a non-empty str")
        if not isinstance(c["args"], dict):
            raise ValueError("args must be a dict")

    canonical = lambda args: json.dumps(_canon(args), sort_keys=True,
                                        separators=(",", ":"), ensure_ascii=False)
    counts = {}
    allowed = []
    fingerprints = []
    counts_at_call = []
    first_block_index = None
    for i, c in enumerate(calls):
        fp = c["tool"] + "|" + hashlib.sha256(canonical(c["args"]).encode("utf-8")).hexdigest()
        seen = counts.get(fp, 0)
        counts_at_call.append(seen)
        fingerprints.append(fp)
        if seen < threshold:
            counts[fp] = seen + 1
            allowed.append(True)
        else:
            allowed.append(False)
            if first_block_index is None:
                first_block_index = i
    return {
        "allowed": allowed,
        "fingerprints": fingerprints,
        "counts_at_call": counts_at_call,
        "first_block_index": first_block_index,
    }
