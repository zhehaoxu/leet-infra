import torch


def solve(requests, max_batch_size):
    """FCFS continuous batching with prefill=1 step + gen_len decode steps。"""
    N = requests.shape[0]
    finish = torch.full((N,), -1, dtype=torch.long)
    arrivals = requests[:, 0].tolist()
    gens = requests[:, 1].tolist()
    waiting = list(range(N))  # 按 (arrival, id) 天然有序（要求 arrival 非降序）
    running = []  # list of [rid, rem]
    step = 0
    while waiting or running:
        cap = int(max_batch_size) - len(running)
        if cap > 0:
            admitted = []
            for i in list(waiting):
                if arrivals[i] <= step:
                    admitted.append(i)
                    if len(admitted) >= cap:
                        break
            for i in admitted:
                waiting.remove(i)
                running.append([i, gens[i] + 1])
        next_running = []
        for (rid, rem) in running:
            rem -= 1
            if rem == 0:
                finish[rid] = step
            else:
                next_running.append([rid, rem])
        running = next_running
        step += 1
    return finish
