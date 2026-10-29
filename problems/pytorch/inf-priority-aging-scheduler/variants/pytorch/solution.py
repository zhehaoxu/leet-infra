def solve(arrivals, service, priority, max_batch, aging_interval, priority_cap):
    n = len(arrivals)
    admit = [-1] * n
    finish = [-1] * n
    waiting = set(range(n))
    running = {}
    done = 0
    t = 0

    while done < n:
        slots = max_batch - len(running)
        while slots > 0:
            best = None
            for i in waiting:
                if arrivals[i] > t:
                    continue
                eff = min(priority_cap, priority[i] + (t - arrivals[i]) // aging_interval)
                key = (-eff, -priority[i], arrivals[i], i)
                if best is None or key < best[0]:
                    best = (key, i)
            if best is None:
                break
            _, i = best
            waiting.discard(i)
            admit[i] = t
            running[i] = service[i]
            slots -= 1

        if not running:
            t = max(t, min(arrivals[i] for i in waiting))
            continue

        finished = []
        for i in running:
            running[i] -= 1
            if running[i] == 0:
                finish[i] = t
                finished.append(i)
        for i in finished:
            del running[i]
            done += 1
        t += 1

    return admit, finish, max(admit[i] - arrivals[i] for i in range(n))
