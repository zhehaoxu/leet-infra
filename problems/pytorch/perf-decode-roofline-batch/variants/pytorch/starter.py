def solve(n_params, layers, kv_heads, head_dim, context_tokens, batch,
          peak_flops, bandwidth, weight_bytes=2, kv_bytes=2,
          target_tokens_per_s=0.0, max_batch=100000):
    """解码 Roofline 的 batch 账：权重读一次、KV 每 token 读一份，求拐点 batch 与 SLO 最小 batch。

    n_params          : 模型参数量 N（整数 >=1，一次解码每 token 都要过一遍全部权重）
    layers            : Transformer 层数 L
    kv_heads          : KV 头数（GQA 时小于 Q 头数）
    head_dim          : 每头维度 d
    context_tokens    : 当前上下文长度 T（已缓存的 token 数，可为 0 表示无 KV）
    batch             : 解码并发的序列数 B（整数 >=1）
    peak_flops        : 单卡峰值算力（FLOP/秒，>0）
    bandwidth            : HBM 带宽（字节/秒，>0）
    weight_bytes      : 权重每参数字节（默认 2=fp16/bf16，可取 1=fp8、4=fp32）
    kv_bytes          : KV 每元素字节（默认 2）
    target_tokens_per_s : 目标吞吐（tokens/s，>=0；0 表示没有 SLO 要求）
    max_batch         : batch 搜索上限（整数 >=1，用于 SLO 判定）

    模型（务必按此定义）：
      weight_bytes_step = n_params * weight_bytes                 # 每步都要重读一遍权重
      kv_bytes_per_token = 2 * layers * kv_heads * head_dim * context_tokens * kv_bytes
                # 2 = K 与 V；口径沿用 kv-cache-memory
      bytes_per_step = weight_bytes_step + batch * kv_bytes_per_token   # 每步从 HBM 搬的字节
      flops_per_token = 2 * n_params                               # 2N 口径，沿用 inference-throughput
      flops_per_step  = batch * flops_per_token
      step_time_s = max(bytes_per_step / bandwidth, flops_per_step / peak_flops)
      tokens_per_s = batch / step_time_s
      is_memory_bound = flops_per_step < bytes_per_step * (peak_flops / bandwidth)
                #  Roofline 口径沿用 roofline-arithmetic：相等算 compute-bound，故这里用严格小于
      weight_fraction = weight_bytes_step / bytes_per_step         # 权重占比，batch 越大越被摊薄
      ceiling_tokens_per_s = min(peak_flops / flops_per_token, bandwidth / kv_bytes_per_token)
                # 前者是算力上限，后者是 batch->inf 时 KV 读的上限；kv_bytes_per_token==0 时取算力上限
      balance_batch = 最小的正整数 B 使该 batch 起进入 compute-bound（即
                      B*(2*n_params*bandwidth - kv_bytes_per_token*peak_flops)
                          >= peak_flops*weight_bytes_step）；
                      若系数 <=0（KV 读永远压不住）则记 -1
      min_batch_for_slo = 在 [1, max_batch] 内使 tokens_per_s >= target_tokens_per_s 的最小 B，
                      找不到记 -1（tokens_per_s 对 B 单调不减，可二分）

    返回 (tokens_per_s, is_memory_bound, step_time_ms, ceiling_tokens_per_s,
          balance_batch, min_batch_for_slo, weight_fraction, flops_per_token)：
      第 1、3、4、7 项 float（atol=1e-5, rtol=1e-5），第 2 项 bool，第 5、6、8 项 int。
    退化：context_tokens=0 -> kv_bytes_per_token=0、bytes 只剩权重、ceiling=peak_flops/(2*n_params)、
    balance_batch=1；batch 远小于 balance_batch 时 tokens_per_s 约等于 batch*bandwidth/weight_bytes_step
    （加 batch 线性提速）；target_tokens_per_s=0 -> min_batch_for_slo=1。
    """
    # TODO
    raise NotImplementedError
