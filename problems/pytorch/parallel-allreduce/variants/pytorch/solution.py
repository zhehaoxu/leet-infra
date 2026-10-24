# 环形 All-Reduce 通信量与耗时参考解
def solve(message_bytes, world, bandwidth_bps, latency_per_hop=0.0):
    S = float(message_bytes)
    W = int(world)
    if W <= 1:
        return (0.0, 0.0)
    bytes_per_rank = 2.0 * (W - 1) / W * S
    hops = 2.0 * (W - 1)
    time = bytes_per_rank / float(bandwidth_bps) + hops * float(latency_per_hop)
    return (bytes_per_rank, time)
