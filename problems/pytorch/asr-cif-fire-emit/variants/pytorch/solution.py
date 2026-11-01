def solve(alphas, hidden, threshold=1.0, tail_threshold=0.45, use_tail=True):
    """FunASR `cif()` fire 发射账（单序列）。

    返回 {"embeds": [...], "fire_positions": [...], "token_num": int}。
    """
    if use_tail:
        # tail_process_fn（mask=None 分支）：alphas 追加 tail_threshold，
        # hidden 追加一行零向量
        alphas = list(alphas) + [tail_threshold]
        hidden = [list(h) for h in hidden] + [[0.0] * len(hidden[0])]

    D = len(hidden[0])
    integrate = 0.0
    frame = [0.0] * D
    fires = []
    snapshots = []
    for t, alpha in enumerate(alphas):
        # 官方：distribution_completion 在加 alpha 前取（当前 token 缺口）
        completion = 1.0 - integrate
        integrate += alpha
        fires.append(integrate)  # 官方快照在 fire 判定/减 1 之前
        fire = integrate >= threshold
        if fire:
            integrate -= 1.0
            cur = completion
        else:
            cur = alpha
        remainds = alpha - cur
        frame = [frame[d] + cur * hidden[t][d] for d in range(D)]
        snapshots.append(frame[:])
        if fire:
            frame = [remainds * hidden[t][d] for d in range(D)]

    fire_positions = [t for t, f in enumerate(fires) if f >= threshold]
    embeds = [snapshots[t] for t in fire_positions]

    total = sum(alphas)
    if use_tail:
        # 官方 tail_process_fn: token_num = floor(alphas.sum())
        token_num = int(total // 1)
    else:
        # 官方 cif(): len_labels = round(alphas.sum())
        token_num = int(total + 0.5) if total >= 0 else -int(-total + 0.5)
    return {"embeds": embeds, "fire_positions": fire_positions,
            "token_num": token_num}
