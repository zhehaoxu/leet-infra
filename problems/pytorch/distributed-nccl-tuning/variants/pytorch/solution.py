"""Select from a supplied, calibrated latency/bandwidth cost table."""


def solve(message_sizes, latencies_us, bandwidths_gbps):
    latencies = [value for row in latencies_us for value in row]
    bandwidths = [value for row in bandwidths_gbps for value in row]

    thresholds = [[None for _ in range(6)] for _ in range(6)]
    for i in range(6):
        for j in range(i + 1, 6):
            bi, bj = bandwidths[i], bandwidths[j]
            if bi == bj:
                continue
            # Algebraically beta_i - beta_j, without subtracting reciprocals.
            slope_gap = (bj - bi) / (1000.0 * bi * bj)
            crossing = (latencies[j] - latencies[i]) / slope_gap
            if crossing >= 0.0:
                # Normalize signed zero and fill both directions identically.
                thresholds[i][j] = thresholds[j][i] = max(0.0, crossing)

    times_us = []
    best = []
    achieved_gbps = []
    utilization = []
    for size in message_sizes:
        times = [
            latency + size / (1000.0 * bandwidth)
            for latency, bandwidth in zip(latencies, bandwidths)
        ]
        # min preserves the first index on an exact tie; no epsilon is used.
        winner = min(range(6), key=times.__getitem__)
        achieved = size / (1000.0 * times[winner])
        times_us.append(times)
        best.append(winner)
        achieved_gbps.append(achieved)
        utilization.append(achieved / bandwidths[winner])

    return {
        "times_us": times_us,
        "best": best,
        "thresholds_bytes": thresholds,
        "achieved_gbps": achieved_gbps,
        "utilization": utilization,
    }
