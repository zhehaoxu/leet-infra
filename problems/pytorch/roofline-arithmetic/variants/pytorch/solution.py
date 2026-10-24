# Roofline 算术强度与性能上限参考解
def solve(flops, nbytes, peak_flops, bandwidth):
    ai = float(flops) / float(nbytes)
    ridge = float(peak_flops) / float(bandwidth)
    if ai < ridge:                       # memory-bound：带宽限制
        attainable = ai * float(bandwidth)
        is_memory_bound = True
    else:                                # compute-bound：算力限制（含 AI==ridge）
        attainable = float(peak_flops)
        is_memory_bound = False
    return (ai, ridge, attainable, is_memory_bound)
