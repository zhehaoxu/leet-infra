def solve(transitions, match_states, sink_state, vocab, eos_token_id):
    """outlines-core Index::new 的直接移植（token 插入序遍历、LIFO 栈）。"""

    def step(state, byte):
        return transitions.get(state, {}).get(byte, sink_state)

    def byte_class_exists(state):
        return any(step(state, b) != sink_state for b in range(256))

    seen = {0}
    stack = [0]
    finals = set()
    idx = {}
    cache = {}

    while stack:
        current = stack.pop()
        has_valid = False
        if current in match_states:
            finals.add(current)
            has_valid = True

        for token, ids in vocab.items():
            if eos_token_id in ids:
                continue
            nxt = current
            ok = True
            for byte in token.encode("utf-8"):
                nxt = step(nxt, byte)
                if nxt == sink_state:
                    ok = False
                    break
            if not ok:
                continue
            if nxt not in cache:
                cache[nxt] = nxt in match_states or byte_class_exists(nxt)
            if cache[nxt]:
                has_valid = True
                row = idx.setdefault(current, {})
                for tid in ids:
                    row[tid] = nxt
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)

        if not has_valid and current not in match_states:
            missing = []
            for b in range(256):
                if step(current, b) != sink_state:
                    missing.append(chr(b) if b < 128 else "\\x%02x" % b)
            return {
                "incompatible": {"state": current, "missing_tokens": missing}
            }

    for f in finals:
        idx.setdefault(f, {})[eos_token_id] = f

    return {
        "initial_state": 0,
        "final_states": sorted(finals),
        "transitions": idx,
        "incompatible": None,
    }
