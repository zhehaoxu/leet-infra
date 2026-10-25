"""参考答案：解码 Roofline 的 batch 账——权重被摊薄 vs KV 线性增长、拐点 batch 与 SLO 反解。"""

import math


def solve(n_params, layers, kv_heads, head_dim, context_tokens, batch,
          peak_flops, bandwidth, weight_bytes=2, kv_bytes=2,
          target_tokens_per_s=0.0, max_batch=100000):
    weight = float(n_params) * float(weight_bytes)                    # 每步重读一遍权重
    kv = 2.0 * layers * kv_heads * head_dim * context_tokens * kv_bytes   # 每序列每步的 KV 读量
    flops_per_token = 2 * int(n_params)                               # 2N 口径

    def throughput(b):
        step_s = max((weight + b * kv) / bandwidth, b * flops_per_token / peak_flops)
        return b / step_s

    # roofline-arithmetic 的口径：算强度 >= ridge 记 compute-bound，相等也算
    def compute_bound(b):
        return b * flops_per_token * bandwidth >= (weight + b * kv) * peak_flops

    is_memory_bound = not compute_bound(batch)
    step_s = max((weight + batch * kv) / bandwidth, batch * flops_per_token / peak_flops)
    tokens_per_s = batch / step_s
    weight_fraction = weight / (weight + batch * kv) if (weight + batch * kv) > 0.0 else 0.0

    compute_cap = peak_flops / flops_per_token                 # 算力上限（batch 无关）
    mem_cap = bandwidth / kv if kv > 0.0 else float("inf")      # batch->inf 时 KV 读的上限
    ceiling = compute_cap if compute_cap < mem_cap else mem_cap

    # B*(2N*bw - K*peak) >= peak*W：系数 <=0 说明加多大 batch 都填不满算力
    denom = 2.0 * float(n_params) * bandwidth - kv * peak_flops
    if denom <= 0.0:
        balance = -1
    else:
        balance = int(math.ceil(peak_flops * weight / denom))
        if balance < 1:
            balance = 1
        while balance > 1 and compute_bound(balance - 1):      # 浮点比值可能差一个 ulp，用判据校正
            balance -= 1
        while not compute_bound(balance):
            balance += 1

    if throughput(max_batch) < target_tokens_per_s:            # tokens_per_s 对 B 单调不减 -> 二分
        min_batch = -1
    else:
        lo, hi = 1, int(max_batch)
        while lo < hi:
            mid = (lo + hi) // 2
            if throughput(mid) >= target_tokens_per_s:
                hi = mid
            else:
                lo = mid + 1
        min_batch = lo

    return (float(tokens_per_s), bool(is_memory_bound), float(step_s * 1000.0), float(ceiling),
            int(balance), int(min_batch), float(weight_fraction), int(flops_per_token))
