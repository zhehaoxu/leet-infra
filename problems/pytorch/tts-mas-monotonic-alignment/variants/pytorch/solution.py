NEG_INF = float("-inf")


def solve(value):
    if not isinstance(value, (list, tuple)) or len(value) == 0:
        raise ValueError("value must be a non-empty [T_mel, T_text] list")
    T_text = None
    for row in value:
        if not isinstance(row, (list, tuple)) or len(row) == 0:
            raise ValueError("value rows must be non-empty lists")
        if T_text is None:
            T_text = len(row)
        elif len(row) != T_text:
            raise ValueError("value rows must have equal length")
        for x in row:
            if isinstance(x, bool) or not isinstance(x, (int, float)):
                raise ValueError("value entries must be numbers")
            if isinstance(x, float) and (x != x or x in (float("inf"), float("-inf"))):
                raise ValueError("value entries must be finite")
    T_mel = len(value)
    if T_mel < T_text:
        raise ValueError("each text token needs at least one mel frame")

    # dp 与 parent 回溯表
    dp = [[NEG_INF] * T_text for _ in range(T_mel)]
    parent = [[-1] * T_text for _ in range(T_mel)]  # 0: 来自上(延续) 1: 来自左上(切换)
    dp[0][0] = value[0][0]
    for t in range(1, T_mel):
        for s in range(T_text):
            best = dp[t - 1][s]
            par = 0
            if s > 0 and dp[t - 1][s - 1] > best:
                best = dp[t - 1][s - 1]
                par = 1
            if best == NEG_INF:
                continue
            dp[t][s] = best + value[t][s]
            parent[t][s] = par

    if dp[T_mel - 1][T_text - 1] == NEG_INF:
        raise ValueError("no monotonic alignment covers every token")

    # 回溯：标记 attn 并统计 durations
    attn = [[0] * T_text for _ in range(T_mel)]
    durations = [0] * T_text
    t, s = T_mel - 1, T_text - 1
    while t >= 0:
        attn[t][s] = 1
        durations[s] += 1
        par = parent[t][s]
        if t == 0:
            break
        if par == 1:
            s -= 1
        t -= 1
    return {"log_prob": dp[T_mel - 1][T_text - 1],
            "attn": attn, "durations": durations}
