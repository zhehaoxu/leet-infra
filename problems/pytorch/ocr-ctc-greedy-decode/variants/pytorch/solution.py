def solve(logits, blank_id=0):
    if not isinstance(logits, (list, tuple)) or len(logits) == 0:
        raise ValueError("logits must be a non-empty [T, V] list")
    V = None
    for row in logits:
        if not isinstance(row, (list, tuple)) or len(row) == 0:
            raise ValueError("logits rows must be non-empty lists")
        if V is None:
            V = len(row)
        elif len(row) != V:
            raise ValueError("logits rows must have equal length")
        for x in row:
            if isinstance(x, bool) or not isinstance(x, (int, float)):
                raise ValueError("logits entries must be numbers")
            if isinstance(x, float) and (x != x or x in (float("inf"), float("-inf"))):
                raise ValueError("logits entries must be finite")
    if isinstance(blank_id, bool) or not isinstance(blank_id, int):
        raise ValueError("blank_id must be an int")
    if not (0 <= blank_id < V):
        raise ValueError("blank_id out of range")

    # 1) argmax per frame, ties pick the smallest class id
    raw_path = []
    for row in logits:
        best = 0
        for c in range(1, V):
            if row[c] > row[best]:
                best = c
        raw_path.append(best)

    # 2) run-collapse on the raw path (blank participates in adjacency),
    #    3) drop blank runs, 4) closed-frame spans
    tokens = []
    spans = []
    T = len(raw_path)
    i = 0
    while i < T:
        a = raw_path[i]
        j = i
        while j + 1 < T and raw_path[j + 1] == a:
            j += 1
        if a != blank_id:
            tokens.append(a)
            spans.append([i, j])
        i = j + 1
    return {"raw_path": raw_path, "tokens": tokens, "spans": spans}
