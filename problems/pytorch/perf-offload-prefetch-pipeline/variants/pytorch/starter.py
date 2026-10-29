def solve(compute, transfer_bytes, bw, buffers):
    """Offload 预取流水线：HBM↔DRAM 双缓冲下「计算-传输」重叠的时间线账。

    compute        : list[int]，逐层的计算耗时（>=1，单位时间）
    transfer_bytes : list[int]，逐层需要从 DRAM 预取到 HBM 的参数字节（>=0）
    bw             : int，预取引擎带宽（字节/单位时间，>=1）
    buffers        : int，HBM 上的预取缓冲槽数（>=1）

    模型（务必按此定义；单引擎、严格按层序预取、计算与传输可同时进行）：
      transfer_time[j] = ceil(transfer_bytes[j] / bw)      # 0 字节记 0，不占引擎
      槽位            : 第 j 层占用槽 j % buffers，从**预取开始**一直占到**计算结束**
      预取开始 s_j    = max(engine_free, finish[j - buffers])   # j < buffers 时后者记 0
                        engine_free 是上一层预取的结束时刻（单引擎串行）
      预取结束 e_j    = s_j + transfer_time[j]
      计算开始 c_j    = max(finish[j - 1], e_j)             # 首层 finish[-1] 记 0
      finish[j]      = c_j + compute[j]
      stall（空等）    = Σ (c_j - finish[j-1])，即计算被预取拖住的单位时间
      makespan       = finish[L-1] = Σ compute + stall

    返回 (makespan, stall_total, compute_start, transfer_start, transfer_end)：
      前两项为 int；后三项为长度 L 的 list[int]，按层序给出各时间点。
    """
    # TODO
    raise NotImplementedError
