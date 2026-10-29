def solve(param_sizes, bucket_cap_mb, world_size, ready_times):
    """Static reverse-order packing and globally ordered readiness eligibility."""
    cap = int(bucket_cap_mb * 1024**2)
    buckets = []
    current = []
    used = 0
    for index in range(len(param_sizes) - 1, -1, -1):
        size = param_sizes[index]
        if current and used + size > cap:
            buckets.append(current)
            current = []
            used = 0
        current.append(index)
        used += size
        if size > cap:
            buckets.append(current)
            current = []
            used = 0
    if current:
        buckets.append(current)

    bucket_ready_times = [
        [max(ready_times[rank][index] for index in bucket) for bucket in buckets]
        for rank in range(world_size)
    ]
    global_ready_times = [
        max(bucket_ready_times[rank][bucket] for rank in range(world_size))
        for bucket in range(len(buckets))
    ]
    enqueue_times = []
    eligible = 0.0
    for ready in global_ready_times:
        eligible = max(eligible, ready)
        enqueue_times.append(eligible)

    return {
        'buckets': buckets,
        'bucket_ready_times': bucket_ready_times,
        'global_ready_times': global_ready_times,
        'enqueue_times': enqueue_times,
    }
