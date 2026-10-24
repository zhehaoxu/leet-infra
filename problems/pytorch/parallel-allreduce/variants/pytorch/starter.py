# 环形 All-Reduce：每卡通信量与耗时
def solve(message_bytes, world, bandwidth_bps, latency_per_hop=0.0):
    # TODO: bytes_per_rank = 2*(W-1)/W*S; time = bytes/bw + 2*(W-1)*latency; W<=1 => (0,0)
    raise NotImplementedError
