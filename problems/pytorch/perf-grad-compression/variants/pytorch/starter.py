def solve(shapes, world, method="sign", bytes_per_element=4, bits_per_element=1,
          scale_group_size=0, rank=1, error_feedback=True):
    """梯度压缩带宽账：1-bit（sign）量化 vs PowerSGD 低秩投影，含 error feedback 与 fp32 回退切换点。

    shapes            : [[rows, cols], ...] 通信矩阵清单（一次 all-reduce 要同步的全部梯度分块）
    world             : 数据并行 rank 数 W（>=1）
    method            : "full" | "sign" | "lowrank"
    bytes_per_element : 未压缩梯度每元素字节（默认 4=fp32；1=fp8、2=fp16 也算）
    bits_per_element  : "sign" 每元素位数（>=1）
    scale_group_size  : "sign" 的分组量化粒度 g（每 g 个元素共享一个 fp32 scale=4 字节）；
                        g=0 表示整块张量共用一个 scale
    rank              : "lowrank" 的秩 r（每个矩阵发 P:(rows,r) 与 Q:(r,cols) 两块）
    error_feedback    : 是否维护本地 fp32 残差累加器（1-bit Adam / PowerSGD 都要求）

    每步**聚合载荷** payload（字节，整数向上取整，空矩阵 numels==0 直接跳过）：
      full    : payload = total_elements * bytes_per_element
      sign    : 每矩阵 payload += ceil(numels*bits/8) + ceil(numels/g)*4（g=0 时 scale 项取 4）
      lowrank : rank >= min(rows,cols) 时该矩阵退化为满秩 -> payload += numels*bytes_per_element；
                否则 payload += (rows + cols) * rank * bytes_per_element
    回退：compressed = min(payload, full_payload)；fallback = (payload > full_payload)
         （压缩反而更贵就切回未压缩的 fp32 all-reduce）
    环系数沿用 distributed-zero-communication：c = (W-1)/W，做一次 all-reduce 每 rank 搬 2*c*S
      per_gpu_comm_bytes = 2*c*compressed ;  full_comm_bytes = 2*c*full_payload
      saved_ratio = 1 - compressed/full_payload（full_payload==0 时记 0.0）
    ef_state_bytes = total_elements*4（error_feedback=False 时为 0）
    max_bits：method=="sign" 时，保持 payload <= full_payload 的最大位数
      b ∈ [1, bytes_per_element*8]（线性扫描；一个都不可行则为 0，full_payload==0 也记 0）；
      其它 method 恒为 0。

    返回 (full_payload, compressed_payload, per_gpu_comm_bytes, full_comm_bytes,
          saved_ratio, ef_state_bytes, fallback, max_bits)：
    第 1、2、6、8 项 int，第 3、4、5 项 float（atol=1e-5, rtol=1e-5），第 7 项 bool。
    退化：W==1 -> 两个 comm 量都是 0.0；method=="full" -> compressed==full、saved_ratio==0.0；
    lowrank 的 rank==min(rows,cols) -> 该矩阵按满秩计（等于 full）。
    """
    # TODO
    raise NotImplementedError
