def solve(probs, beam_width=3, blank_id=0, k=None):
    if not isinstance(probs, (list, tuple)) or len(probs) == 0:
        raise ValueError("probs must be a non-empty [T, V] list")
    V = None
    for row in probs:
        if not isinstance(row, (list, tuple)) or len(row) == 0:
            raise ValueError("probs rows must be non-empty lists")
        if V is None:
            V = len(row)
        elif len(row) != V:
            raise ValueError("probs rows must have equal length")
        for x in row:
            if isinstance(x, bool) or not isinstance(x, (int, float)):
                raise ValueError("probs entries must be numbers")
            if x < 0:
                raise ValueError("probs entries must be non-negative")
            if isinstance(x, float) and (x != x or x in (float("inf"), float("-inf"))):
                raise ValueError("probs entries must be finite")
    if isinstance(blank_id, bool) or not isinstance(blank_id, int) \
            or not (0 <= blank_id < V):
        raise ValueError("blank_id out of range")
    if isinstance(beam_width, bool) or not isinstance(beam_width, int) or beam_width < 1:
        raise ValueError("beam_width must be a positive int")
    if k is not None:
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive int or None")

    T = len(probs)
    beam = {(): [1.0, 0.0]}  # prefix -> [p_blank_end, p_nonblank_end]
    for t in range(T):
        new = {}

        def slot(s):
            if s not in new:
                new[s] = [0.0, 0.0]
            return new[s]

        for prefix, (pb, pnb) in beam.items():
            row = slot(prefix)
            row[0] += (pb + pnb) * probs[t][blank_id]
            for c in range(V):
                if c == blank_id:
                    continue
                p = probs[t][c]
                if p == 0.0:
                    continue
                if prefix and prefix[-1] == c:
                    slot(prefix + (c,))[1] += pb * p   # 新字符只能来自 blank 结尾
                    row[1] += pnb * p                  # 重复延续，前缀不变
                else:
                    slot(prefix + (c,))[1] += (pb + pnb) * p
        items = sorted(new.items(),
                       key=lambda kv: (-(kv[1][0] + kv[1][1]), kv[0]))
        beam = dict(items[:beam_width])

    out = sorted(beam.items(),
                 key=lambda kv: (-(kv[1][0] + kv[1][1]), kv[0]))
    kk = beam_width if k is None else k
    return {"hypotheses": [{"prefix": list(p), "prob": pb + pnb}
                           for p, (pb, pnb) in out[:kk] if pb + pnb > 0.0]}
