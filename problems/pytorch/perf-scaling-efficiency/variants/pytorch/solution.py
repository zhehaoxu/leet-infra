"""参考答案：按「整机吞吐」统一定义求加速比与效率，饱和点用吞吐比较。"""

import torch


def solve(world_sizes, step_times_ms, mode):
    w = [float(x) for x in world_sizes]
    t = [float(x) for x in step_times_ms]
    n = len(w)

    if mode == "strong":        # 每步总样本数固定：吞吐只看每步耗时
        throughput = [1.0 / x for x in t]
    else:                       # weak：每卡样本数固定，每步总样本正比于 w
        throughput = [w[i] / t[i] for i in range(n)]

    base = throughput[0]
    speedup = [throughput[i] / base for i in range(n)]
    efficiency = [speedup[i] / (w[i] / w[0]) for i in range(n)]

    super = [i for i in range(n) if efficiency[i] > 1.0]
    saturation = -1
    for i in range(1, n):
        if throughput[i] <= throughput[i - 1]:
            saturation = i
            break
    return (torch.tensor(speedup, dtype=torch.float64),
            torch.tensor(efficiency, dtype=torch.float64),
            torch.tensor(super, dtype=torch.int64),
            int(saturation))
