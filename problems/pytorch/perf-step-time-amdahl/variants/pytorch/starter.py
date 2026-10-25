def solve(phase_ms, target_phase, speedup, worker_ms, num_workers, prefetch):
    """训练 step 时间分解 + Amdahl 上限 + dataloader 掩盖判定（单位 ms，串行相位口径）。

    phase_ms     : 4 个非负数 [dataloader_wait, forward, backward, optimizer]，profiler 量到的每步墙钟占比
    target_phase : 0..3，被提速的相位下标
    speedup      : k>0，该相位自身耗时除以 k（k=1 表示不改）
    worker_ms    : 单个 dataloader worker 生产一个 batch 的 CPU 耗时
    num_workers  : >=1 的 worker 数
    prefetch     : 预取队列深度，0 表示同步取数（无重叠）

    记 total = sum(phase_ms)，p = phase_ms[target_phase]：
      fracs[i]        = phase_ms[i] / total            # 占比，tensor float64
      new_step_ms     = (total - p) + p / k            # 只有该相位被压缩
      amdahl_speedup  = total / new_step_ms            # new_step_ms==0 时记 inf
      ceiling_speedup = total / (total - p)            # k->无穷 的上限；total==p 时记 inf
    dataloader 侧（与上面的 total 是两套视角，互不混用）：
      supply_ms  = worker_ms / num_workers             # 每 supply_ms 出一个 batch
      compute_ms = phase_ms[1] + phase_ms[2] + phase_ms[3]   # 不含取数的计算时长
      wait_ms    = supply_ms               若 prefetch == 0 否则 max(0, supply_ms - compute_ms)
      effective_step_ms = compute_ms + wait_ms         # 稳态每步摊销（首个 batch 的暖机不计）
      data_hidden       = (wait_ms == 0)               # 恰好跟得上也算 True

    返回 (fracs, total_ms, new_step_ms, amdahl_speedup, ceiling_speedup, effective_step_ms, data_hidden)。
    退化：k=1 时 new_step_ms==total 且 amdahl_speedup==1.0；其余相位全为 0 时 ceiling_speedup 记 inf；
    prefetch=0 时 wait_ms==supply_ms（同步取数，永不掩盖，除非 worker_ms=0）；
    prefetch>=1 且 supply_ms<=compute_ms 时 wait_ms==0、data_hidden 为 True。
    """
    # TODO
    raise NotImplementedError
