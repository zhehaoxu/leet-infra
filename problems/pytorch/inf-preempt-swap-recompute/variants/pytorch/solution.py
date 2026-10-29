def solve(kv, prompt, generated, priority, need, bytes_per_token, swap_bw, tick_per_token, freezable):
    victims = []
    modes = []
    freed = 0
    ticks = 0
    replay = 0

    tiers = sorted({priority[i] for i in range(len(kv)) if freezable[i]})
    for level in tiers:
        if freed >= need:
            break
        candidates = [i for i in range(len(kv)) if freezable[i] and priority[i] == level]
        candidates.sort(key=lambda i: (-kv[i], i))
        for i in candidates:
            if freed >= need:
                break
            swap = -(-(2 * kv[i] * bytes_per_token) // swap_bw)
            recompute = (prompt[i] + generated[i]) * tick_per_token
            if swap <= recompute:
                modes.append("swap")
                ticks += swap
            else:
                modes.append("recompute")
                ticks += recompute
                replay += prompt[i] + generated[i]
            victims.append(i)
            freed += kv[i]

    if freed < need:
        return [], [], freed, -1, 0
    return victims, modes, freed, ticks, replay
