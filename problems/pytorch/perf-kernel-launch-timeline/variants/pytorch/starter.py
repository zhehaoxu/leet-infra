def solve(durations_us, launch_overhead_us, use_graph):
    """Kernel 发射时间线：busy/idle 账与 CUDA Graph 收益（单条串行流，时间单位微秒 us）。

    durations_us       : 列表，第 i 个 kernel 的 GPU 执行时长 d[i]（float，>=0）
    launch_overhead_us : 每次发射（launch）的 CPU 启动开销 L（float，>=0）
    use_graph          : True 表示整串 kernel 被捕获成一张 CUDA Graph、replay 只发射一次

    eager 时间线（每 kernel 各发一次）：
      r_i = (i + 1) * L                       # 第 i 个 kernel 的发射就绪时刻（CPU 从 0 起连续发射）
      e_{-1} = 0 ; s_i = max(r_i, e_{i-1}) ; e_i = s_i + d[i]
    graph 时间线（一次发射放行全部 kernel，背靠背执行）：
      graph_wall = L + sum(d)

    返回六元组 (eager_wall_us, graph_wall_us, wall_us, busy_ratio, launch_bound_count, speedup)：
      eager_wall_us      = e_{n-1}
      graph_wall_us      = L + sum(d)
      wall_us            = graph_wall_us 若 use_graph 否则 eager_wall_us
      busy_ratio         = sum(d) / wall_us        （wall_us == 0 时记 1.0）
      launch_bound_count = eager 时间线中「GPU 因等发射而空转」的 kernel 个数，即满足 r_i > e_{i-1}
                           的 i 的个数（是 eager 的属性，与 use_graph 无关）
      speedup            = eager_wall_us / graph_wall_us （graph_wall_us == 0 时记 1.0，恒 >= 1）

    退化：n=1 或 L=0 时 eager_wall == graph_wall（speedup=1.0）；L=0 时 launch_bound_count=0；
    全部 d=0 时 eager_wall = n*L、graph_wall = L、busy_ratio=0（wall>0 时）。
    """
    # TODO
    raise NotImplementedError
