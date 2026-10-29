"""参考答案：逐层推进三个时刻（引擎空闲、槽位释放、上一步计算结束）。"""


def solve(compute, transfer_bytes, bw, buffers):
    bw = int(bw)
    buffers = int(buffers)
    layers = len(compute)

    finish = []
    compute_start = []
    transfer_start = []
    transfer_end = []
    engine_free = 0
    stall = 0

    for j in range(layers):
        nbytes = int(transfer_bytes[j])
        transfer_time = -(-nbytes // bw) if nbytes > 0 else 0
        slot_ready = finish[j - buffers] if j >= buffers else 0
        s = max(engine_free, slot_ready)
        e = s + transfer_time
        engine_free = e
        prev_finish = finish[j - 1] if j > 0 else 0
        c = max(prev_finish, e)
        stall += c - prev_finish
        compute_start.append(int(c))
        transfer_start.append(int(s))
        transfer_end.append(int(e))
        finish.append(c + int(compute[j]))

    return (
        int(finish[-1]) if finish else 0,
        int(stall),
        compute_start,
        transfer_start,
        transfer_end,
    )
