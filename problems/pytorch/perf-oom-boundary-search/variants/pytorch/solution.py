"""参考答案：整数字节预算下的两条边界——m 用精确 floor，s 用二次方程正根 + ±1 校正。"""

import math


def solve(hbm_bytes, reserve_bytes, cuda_overhead_bytes, param_bytes, grad_bytes, optim_bytes,
          act_per_token, act_per_token_sq, micro_batch, seq, global_batch, accum_steps):
    usable = int(hbm_bytes) - int(reserve_bytes)
    static = int(param_bytes) + int(grad_bytes) + int(optim_bytes)
    overhead = int(cuda_overhead_bytes)
    ap = int(act_per_token)
    apq = int(act_per_token_sq)
    m = int(micro_batch)
    s = int(seq)

    def peak(mm, ss):
        return overhead + static + mm * ss * ap + mm * ss * ss * apq

    def fits(mm, ss):
        return peak(mm, ss) <= usable

    peak_bytes = peak(m, s)
    headroom = usable - peak_bytes
    feasible = peak_bytes <= usable
    budget = usable - overhead - static          # 能分给激活的字节

    if budget < 0:                               # 连固定开销都放不下
        max_micro, max_seq = 0, 0
    else:
        per_micro = s * ap + s * s * apq         # 每多一个 micro batch 的成本（对 m 线性）
        max_micro = budget // per_micro          # 精确 floor，不需要搜索
        if apq == 0:                             # 只有线性项 -> 同样是 floor
            cand = budget // (m * ap)
        else:                                    # m*apq*s^2 + m*ap*s <= budget 的正根
            a = m * apq
            b = m * ap
            cand = int((-b + math.sqrt(b * b + 4.0 * a * budget)) / (2.0 * a))
        while fits(m, cand + 1):                 # 浮点开方可能差一点点 -> 用判据校正
            cand += 1
        while cand > 0 and not fits(m, cand):
            cand -= 1
        max_seq = cand if cand >= 1 else 0

    per_rank = int(accum_steps) * m              # 一个 rank 一步能吃的序列数
    world = (int(global_batch) + per_rank - 1) // per_rank
    global_fits = bool(feasible and world * per_rank == int(global_batch))

    return (int(static), int(peak_bytes), int(headroom), bool(feasible), int(max_micro),
            int(max_seq), int(world), global_fits)
