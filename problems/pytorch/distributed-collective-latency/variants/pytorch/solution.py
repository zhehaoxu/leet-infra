import math


def solve(W: int, S: float, alpha: float, beta: float) -> dict:
    ring_time = 2 * (W - 1) * (alpha + (S / W) * beta)
    rd_time = math.log2(W) * (alpha + S * beta)
    faster = "ring" if ring_time <= rd_time else "recursive_doubling"
    speedup = max(ring_time, rd_time) / min(ring_time, rd_time)
    return {
        "ring_time": ring_time,
        "rd_time": rd_time,
        "faster": faster,
        "speedup": speedup,
    }
