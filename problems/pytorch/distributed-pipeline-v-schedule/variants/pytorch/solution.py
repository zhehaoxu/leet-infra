def solve(n_stages, n_microbatches, n_virtual, n_layers, act_bytes):
    p, m, v = n_stages, n_microbatches, n_virtual
    chunks = p * v
    if n_layers % chunks:
        raise ValueError("n_layers must be divisible by n_stages * n_virtual")
    owners = [c % p if (c // p) % 2 == 0 else p - 1 - c % p for c in range(chunks)]
    local_chunks = [[c for c in range(chunks) if owners[c] == r] for r in range(p)]
    forward = [0] * chunks
    backward = [0] * chunks
    live = [0] * p
    peak = [0] * p
    timeline = [[] for _ in range(p)]
    while backward[0] < m:
        chosen = [None] * p
        for r, local in enumerate(local_chunks):
            ready_b = [c for c in local if backward[c] < forward[c]
                       and (c == chunks - 1 or backward[c + 1] > backward[c])]
            if ready_b:
                c = min(ready_b, key=lambda c: (backward[c], -c))
                chosen[r] = ("B", c, backward[c])
            else:
                ready_f = [c for c in local if forward[c] < m
                           and (c == 0 or forward[c - 1] > forward[c])]
                if ready_f:
                    c = min(ready_f, key=lambda c: (forward[c], c))
                    chosen[r] = ("F", c, forward[c])
        for r, op in enumerate(chosen):
            timeline[r].append(op)
            if op is not None:
                kind, c, _ = op
                if kind == "F":
                    forward[c] += 1
                    live[r] += 1
                    peak[r] = max(peak[r], live[r])
                else:
                    backward[c] += 1
                    live[r] -= 1
    makespan = len(timeline[0])
    util = 2 * m * chunks / (p * makespan)
    chunk_bytes = (n_layers // chunks) * float(act_bytes)
    peak_bytes = [x * chunk_bytes for x in peak]
    flush = [m * v * chunk_bytes for _ in range(p)]
    return {
        "timeline": timeline,
        "busy_table": [[op is not None for op in row] for row in timeline],
        "makespan": makespan,
        "bubble": 1.0 - util,
        "util": util,
        "peak_activation_bytes": peak_bytes,
        "flush_peak_activation_bytes": flush,
        "memory_saved_bytes": sum(flush) - sum(peak_bytes),
    }
