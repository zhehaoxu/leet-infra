def solve(world_sizes, step_times_ms, mode):
    """强/弱扩展的 scaling efficiency 账：加速比、效率、超线性点与饱和点。

    world_sizes   : 升序正整数列表 w[i]，第 i 档配置的 GPU 卡数；第 0 档是基线
    step_times_ms : 与 world_sizes 等长的正数列表 t[i]，该档**每步墙钟耗时**（ms）
    mode          : "strong"（总工作量固定 -> 整机吞吐按 1/t 计）
                    "weak"  （每卡工作量固定 -> 每步样本数正比 w -> 整机吞吐按 w/t 计）

    定义（第 0 档为基线）：
      throughput[0]   = 1/t[0]  (strong)  /  w[0]/t[0]  (weak)
      speedup[i]      = throughput[i] / throughput[0]      # 恒有 speedup[0] = 1.0
      efficiency[i]   = speedup[i] / (w[i] / w[0])         # 恒有 efficiency[0] = 1.0
                      # 等价闭式：strong -> (t[0]/t[i]) * (w[0]/w[i]) ; weak -> t[0]/t[i]
      superlinear     : efficiency[i] > 1.0 的下标 i（严格大于，升序 int64 张量，可为空）
      saturation_index: 最小的 i >= 1 使 throughput[i] <= throughput[i-1]（再加卡吞吐不再上升）；
                        整条曲线严格上升时返回 -1

    返回四元组 (speedup, efficiency, superlinear, saturation_index)：
    前两项是 float64 张量（与输入等长），第三项 int64 张量，第四项 int。
    退化：只有一档 -> 两个 [1.0]、空 superlinear、saturation_index=-1；
    完美强扩展 t[i] = t[0]*w[0]/w[i] -> speedup = w/w[0]、efficiency 恒 1.0。
    """
    # TODO
    raise NotImplementedError
