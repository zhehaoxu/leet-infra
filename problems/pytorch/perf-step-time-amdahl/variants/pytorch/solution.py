"""参考答案：按题面定义直接算 Amdahl 账与 dataloader 掩盖判据。"""

import torch


def solve(phase_ms, target_phase, speedup, worker_ms, num_workers, prefetch):
    p = [float(x) for x in phase_ms]
    total = sum(p)
    fracs = torch.tensor([x / total for x in p], dtype=torch.float64)

    phase = p[int(target_phase)]
    k = float(speedup)
    rest = total - phase                       # 未改动相位的总时长
    new_step = rest + phase / k
    amdahl = total / new_step if new_step > 0.0 else float("inf")
    ceiling = total / rest if rest > 0.0 else float("inf")

    supply = float(worker_ms) / float(num_workers)
    compute = p[1] + p[2] + p[3]
    if int(prefetch) == 0:                     # 同步取数：完全串进关键路径，无重叠
        wait = supply
    else:                                      # 有预取队列：稳态只补「供不上」的那部分
        wait = supply - compute if supply > compute else 0.0
    effective = compute + wait
    hidden = wait == 0.0
    return (fracs, float(total), float(new_step), float(amdahl),
            float(ceiling), float(effective), bool(hidden))
