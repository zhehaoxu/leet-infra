"""参考答案：异步 rollout 的准入-截断-成批-陈旧度账（纯整数事件模拟，不生成任何 token）。"""

import torch


def solve(gen_lens, arrivals, capacity, step_time, batch_size, max_staleness):
    lens = gen_lens.reshape(-1).to(torch.long)
    arr = arrivals.reshape(-1).to(torch.long)
    n = int(lens.numel())
    c = int(capacity)
    st = int(step_time)
    b = int(batch_size)
    s = int(max_staleness)
    if n == 0:
        empty = torch.zeros(0, dtype=torch.long)
        return empty, empty.clone(), empty.clone(), empty.clone(), empty.clone()

    # 1) FCFS 准入：按 (到达时间, 请求号) 排队，每个请求占用「最早空出来」的那个槽
    order = sorted(range(n), key=lambda i: (int(arr[i]), i))
    start = torch.zeros(n, dtype=torch.long)
    finish = torch.zeros(n, dtype=torch.long)
    truncated = torch.zeros(n, dtype=torch.long)
    release = [0] * c                            # 每个生成槽下一次空闲的时刻
    for pos in range(n):
        i = int(order[pos])
        j = 0
        for k in range(1, c):                    # 取最早空闲的槽；并列取槽号最小（槽等价）
            if release[k] < release[j]:
                j = k
        t0 = max(int(arr[i]), release[j])
        v0 = t0 // st                            # 入场时生效的策略版本
        deadline = (v0 + s) * st                 # partial rollout 的硬截止
        t1 = min(t0 + int(lens[i]), deadline)
        release[j] = t1
        start[i] = t0
        finish[i] = t1
        truncated[i] = 1 if t1 < t0 + int(lens[i]) else 0

    # 2) 按完成时间成批：(finish, index) 升序名次 -> batch_id = 名次 // batch_size
    rank = torch.zeros(n, dtype=torch.long)
    fin_list = finish.tolist()
    for i in range(n):
        r = 0
        for j in range(n):
            if fin_list[j] < fin_list[i] or (fin_list[j] == fin_list[i] and j < i):
                r += 1
        rank[i] = r
    batch_id = rank // b
    num_batches = (n + b - 1) // b

    # 3) 每批的可用时刻 = 组内最晚的完成时间；消费版本 = 该时刻所在的网格版本
    ready = torch.zeros(num_batches, dtype=torch.long)
    for k in range(num_batches):
        member = batch_id == k
        ready[k] = int(finish[member].max())
    consume = torch.zeros(num_batches, dtype=torch.long)
    for k in range(num_batches):
        consume[k] = int(ready[k]) // st

    # 4) 陈旧度与空转
    mu = consume[batch_id] - start // st
    stall = (consume - torch.arange(num_batches, dtype=torch.long)) * st
    stall = torch.clamp(stall, min=0)

    # 5) 队列占用时间线：occupancy[tau] = #{i: start_i <= tau < finish_i}
    horizon = int(finish.max())
    occupancy = torch.zeros(horizon, dtype=torch.long)
    for tau in range(horizon):
        occupancy[tau] = int(((start <= tau) & (tau < finish)).sum())

    return mu.to(torch.long), batch_id.to(torch.long), truncated.to(torch.long), \
        occupancy.to(torch.long), stall.to(torch.long)
