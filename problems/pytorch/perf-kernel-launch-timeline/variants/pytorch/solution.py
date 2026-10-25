"""参考答案：eager 用逐步递推（就绪时刻 vs 上一个完成时刻），graph 用一次发射的闭合式。"""


def solve(durations_us, launch_overhead_us, use_graph):
    d = [float(x) for x in durations_us]
    L = float(launch_overhead_us)
    work = sum(d)

    prev_end = 0.0
    eager_wall = 0.0
    launch_bound = 0
    for i, di in enumerate(d):
        ready = (i + 1) * L
        if ready > prev_end:          # GPU 干等 CPU 发射 -> 空转
            launch_bound += 1
        start = ready if ready > prev_end else prev_end
        prev_end = start + di
        eager_wall = prev_end

    graph_wall = L + work             # 一次 replay 放行全部 kernel，之后背靠背
    wall = graph_wall if use_graph else eager_wall
    busy = work / wall if wall > 0.0 else 1.0
    speedup = eager_wall / graph_wall if graph_wall > 0.0 else 1.0
    return (float(eager_wall), float(graph_wall), float(wall),
            float(busy), int(launch_bound), float(speedup))
