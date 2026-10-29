def solve(arrivals, prompt_lens, gen_lens, chunk, max_batch):
    n = len(arrivals)
    order = sorted(range(n), key=lambda i: (arrivals[i], i))
    admitted = []
    prefill_left = {i: prompt_lens[i] for i in range(n)}
    generated = [0] * n
    first = [-1] * n
    finish = [-1] * n
    ptr = 0
    done = 0
    prefill_steps = 0
    decode_steps = 0
    t = 0

    while done < n:
        while len(admitted) < max_batch and ptr < n and arrivals[order[ptr]] <= t:
            admitted.append(order[ptr])
            ptr += 1

        target = next((i for i in admitted if prefill_left[i] > 0), None)
        if target is not None:
            prefill_left[target] = max(0, prefill_left[target] - chunk)
            prefill_steps += 1
            t += 1
            continue

        active = [i for i in admitted if prefill_left[i] == 0]
        if active:
            decode_steps += 1
            for i in active:
                generated[i] += 1
                if generated[i] == 1:
                    first[i] = t
                if generated[i] == gen_lens[i]:
                    finish[i] = t
                    done += 1
            admitted = [i for i in admitted if finish[i] == -1]
            t += 1
            continue

        t = max(t, arrivals[order[ptr]])

    ttft = [first[i] - arrivals[i] for i in range(n)]
    decode_span = [finish[i] - first[i] for i in range(n)]
    return ttft, decode_span, finish, prefill_steps, decode_steps
