def solve(p_cmp_heads, L, l, top_n, window, gates, branch_outputs):
    """NSA 三分支调度账（l = d = l' 简化口径）。"""

    if not isinstance(p_cmp_heads, list) or not p_cmp_heads:
        raise ValueError("empty p_cmp_heads")
    n_blocks = len(p_cmp_heads[0])
    for row in p_cmp_heads:
        if not isinstance(row, list) or len(row) != n_blocks:
            raise ValueError("head score length mismatch")
    if n_blocks != L // l:
        raise ValueError("n_blocks != L // l")
    if top_n < 1 or window < 1:
        raise ValueError("bad top_n or window")
    if len(gates) != 3:
        raise ValueError("gates must have 3 entries")
    o_cmp, o_slc, o_win = branch_outputs
    if not o_cmp or not (len(o_cmp) == len(o_slc) == len(o_win)):
        raise ValueError("branch output length mismatch")

    # 式 (10)：组内求和
    shared = [0.0] * n_blocks
    for row in p_cmp_heads:
        for i, s in enumerate(row):
            shared[i] += s

    # 式 (11)：降序 rank，并列按块索引升序；rank <= top_n 入选
    order = sorted(range(n_blocks), key=lambda i: (-shared[i], i))
    selected = sorted(order[: min(top_n, n_blocks)])

    # 式 (6) 各分支 size 项
    cmp_len = n_blocks * l
    slc_len = len(selected) * l
    win_len = min(window, L)

    # 式 (5)：门控聚合
    g_cmp, g_slc, g_win = gates
    out = [
        g_cmp * a + g_slc * b + g_win * c
        for a, b, c in zip(o_cmp, o_slc, o_win)
    ]

    return {
        "shared_scores": shared,
        "selected_blocks": selected,
        "cmp_kv_len": cmp_len,
        "slc_kv_len": slc_len,
        "win_kv_len": win_len,
        "n_remapped": cmp_len + slc_len + win_len,
        "output": out,
    }
