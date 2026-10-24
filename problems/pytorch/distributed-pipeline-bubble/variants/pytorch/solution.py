def solve(n_stages, n_microbatches):
    p = float(n_stages)
    m = float(n_microbatches)
    makespan = m + p - 1.0
    util = m / makespan
    bubble = (p - 1.0) / makespan
    return (float(bubble), float(util), float(makespan))
