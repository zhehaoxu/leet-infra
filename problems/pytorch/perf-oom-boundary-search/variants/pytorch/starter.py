def solve(hbm_bytes, reserve_bytes, cuda_overhead_bytes, param_bytes, grad_bytes, optim_bytes,
          act_per_token, act_per_token_sq, micro_batch, seq, global_batch, accum_steps):
    """OOM 边界搜索：给定峰值显存公式，反解「还能开多大 micro batch / 还能多长的序列」。

    hbm_bytes          : 单卡显存容量（字节，>0）
    reserve_bytes      : 预留给分配器碎片/CUDA context 之外的量（字节，>=0），**永远不可用**
    cuda_overhead_bytes: CUDA context、cuBLAS workspace 等固定开销（字节，>=0）
    param_bytes        : 本 rank 上的参数占用（字节，>=0）
    grad_bytes         : 本 rank 上的梯度占用（字节，>=0）
    optim_bytes        : 本 rank 上的优化器状态占用（字节，>=0）
    act_per_token      : 每个「micro batch × 序列位置」的线性激活字节（>=1，反向要留的那份）
    act_per_token_sq   : 注意力分数/softmax 缓存的二次项系数（>=0）：
                         每个 micro batch 序列的该项为 s*s*act_per_token_sq
    micro_batch        : 单个 micro batch 的序列数 m（>=1）
    seq                : 序列长度 s（>=1）
    global_batch       : 目标全局 batch G（序列条数，>=1）
    accum_steps        : 每步梯度累积的 micro batch 数 A（>=1）

    模型（务必按此定义）：
      usable        = hbm_bytes - reserve_bytes                      # 可为负
      static_bytes  = param_bytes + grad_bytes + optim_bytes          # ZeRO-0：每卡都存全量
      peak(m, s)    = cuda_overhead_bytes + static_bytes
                      + m * s * act_per_token                          # 线性激活
                      + m * s * s * act_per_token_sq                   # O(s^2) 注意力项
      fits(m, s)    = peak(m, s) <= usable                             # 闭区间：刚好用满算可行
      headroom_bytes = usable - peak(micro_batch, seq)                 # 可为负，不 clamp
      max_micro_batch = 固定 s 时满足 fits 的最大整数 m>=1；连 m=1 都放不下则记 0
                        （该项对 m 线性，可直接整除 floor）
      max_seq         = 固定 m 时满足 fits 的最大整数 s>=1；连 s=1 都放不下则记 0
                        （对 s 是二次不等式 m*apq*s^2 + m*ap*s <= usable-overhead-static，
                         取二次方程正根的 floor，再用 fits 校正 ±1；apq==0 时退化为线性）
      world_size_needed = ceil(global_batch / (accum_steps * micro_batch))
                        # 用**给定的 micro_batch**，不是 max_micro_batch
      global_batch_fits = fits(micro_batch, seq) 且 world_size_needed*accum_steps*micro_batch
                          == global_batch
                        # 数据并行度必须是整数 rank，最后一个小 micro batch 不许半空

    返回 (static_bytes, peak_bytes, headroom_bytes, feasible, max_micro_batch, max_seq,
          world_size_needed, global_batch_fits)：
      第 1、2、3、5、6、7 项 int（精确相等），第 4、8 项 bool。
    退化：reserve 大到 usable < cuda+static 时 feasible=False、headroom<0、两个边界都是 0；
      act_per_token_sq=0（flash attention 只剩线性项）时 max_seq 走线性 floor；
      G 恰好整除 A*m 时 global_batch_fits 只看显存，否则为 False。
    """
    # TODO
    raise NotImplementedError
