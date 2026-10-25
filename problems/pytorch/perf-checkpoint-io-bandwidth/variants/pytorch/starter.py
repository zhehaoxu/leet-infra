def solve(n_params, step_ms, checkpoint_interval, device_to_host_bw, storage_bw,
          async_save=True, staging_buffers=2, mtbf_steps=1e6, state_bytes_per_param=16):
    """Checkpoint I/O 与异步持久化带宽账：状态字节 -> stall、检查点开销占比与最优周期。

    n_params             : 可训练参数量 N（整数）
    step_ms              : 单个训练 step 的墙钟耗时（ms，>0）
    checkpoint_interval  : 每多少个 step 存一次（整数 >=1）
    device_to_host_bw    : HBM -> pinned host memory 的拷贝带宽（字节/秒，>0）
    storage_bw           : 持久存储（并行文件系统 / NVMe）带宽（字节/秒，>0）
    async_save           : True 表示异步落盘（先拷进 pinned staging，写盘与训练重叠）
    staging_buffers      : pinned staging 缓冲个数 B（>=1，双缓冲即 2）
    mtbf_steps           : 平均故障间隔（以 step 计，>0）
    state_bytes_per_param: 每参数训练态字节，默认 16 —— 沿用 training-state-memory 的
                           混合精度 + Adam 口径（fp16 参数2 + fp16 梯度2 + fp32 master4 + m4 + v4）

    模型（务必按此定义）：
      state_bytes = n_params * state_bytes_per_param
      copy_ms  = 1000 * state_bytes / device_to_host_bw     # D2H 拷进 staging，永远阻塞训练
      write_ms = 1000 * state_bytes / storage_bw            # staging 落盘，异步时与训练重叠
      period_ms = checkpoint_interval * step_ms
      stall_ms  = copy_ms + write_ms          （async_save=False，全同步）
                = copy_ms                     （async_save=True，只付阻塞的那次拷贝）
      overhead_fraction = stall_ms / period_ms
      writer_limited    = (copy_ms + write_ms) > staging_buffers * period_ms
                # 槽位从拷贝起就被占用到落盘结束；B 个槽位可让一次检查点落后 B 个周期，
                # 再慢就必然要等位。同步/异步都用这一判据（同步时它只是参考指标）。
      optimal_interval_steps = floor(sqrt(2 * (stall_ms / step_ms) * mtbf_steps))，最小取 1
                # Young-Daly / CheckFreq 的开销-重算折中：C 为「每次检查点阻塞的 step 数」

    返回 (state_bytes, copy_ms, write_ms, stall_ms, period_ms, overhead_fraction,
          writer_limited, optimal_interval_steps)：
      第 1 项 int，第 2-6 项 float（atol=1e-5, rtol=1e-5），第 7 项 bool，第 8 项 int。
    退化：带宽取极大时 copy/write/stall 趋于 0、overhead 0、writer_limited=False、
    optimal_interval_steps=1（钳位）；async_save=False 时 stall 恒为 copy+write。
    """
    # TODO
    raise NotImplementedError
