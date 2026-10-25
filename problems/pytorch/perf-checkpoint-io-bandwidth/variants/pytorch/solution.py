"""参考答案：状态字节 / 两级带宽 -> 阻塞时长，再加 Young-Daly 的最优检查点周期。"""

import math


def solve(n_params, step_ms, checkpoint_interval, device_to_host_bw, storage_bw,
          async_save=True, staging_buffers=2, mtbf_steps=1e6, state_bytes_per_param=16):
    state_bytes = int(n_params) * int(state_bytes_per_param)
    copy_ms = 1000.0 * state_bytes / float(device_to_host_bw)
    write_ms = 1000.0 * state_bytes / float(storage_bw)
    period_ms = float(checkpoint_interval) * float(step_ms)

    # 异步只付阻塞的那次 D2H 拷贝；同步把落盘也串进关键路径
    stall_ms = copy_ms if async_save else copy_ms + write_ms
    overhead = stall_ms / period_ms

    # 槽位被「拷贝 + 落盘」占用，B 个槽位允许写盘落后 B 个检查点周期
    writer_limited = (copy_ms + write_ms) > float(staging_buffers) * period_ms

    cost_steps = stall_ms / float(step_ms)           # 每次检查点阻塞多少个 step
    optimal = int(math.floor(math.sqrt(2.0 * cost_steps * float(mtbf_steps))))
    if optimal < 1:
        optimal = 1                                  # 至少每步一存
    return (int(state_bytes), float(copy_ms), float(write_ms), float(stall_ms),
            float(period_ms), float(overhead), bool(writer_limited), int(optimal))
