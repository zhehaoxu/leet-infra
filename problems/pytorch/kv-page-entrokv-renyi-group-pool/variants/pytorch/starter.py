import torch


def solve(attn, num_retained_blocks, num_query_heads, num_kv_heads,
          block_size=16, tau_g=0.5, alpha=2.0, sink_size=4):
    """Page-EntroKV 的分组池化 + 页级 top-k：给定注意力权重，选出每个物理 KV head 保留哪些页。

    `attn` 形状 [B, H_Q, T]，非负；组内 r = H_Q // H_KV 个 query head 连续分组。
    对每个 (b, g) 依次：

      1. 熵：把该行前 min(sink_size, T) 个位置置 0（sink 隔离），对剩下的位置重新归一化
         （分母下限 1e-12），再算 Rényi 熵——alpha == 1.0 走 Shannon 熵，
         否则 log(sum p^alpha + 1e-12) / (1 - alpha)。
      2. 组权重：softmax(-ent / tau_g)，只在这组的 r 个 head 上做。
      3. 池化：A_pooled[g][t] = sum_i w_i * attn[b][g*r + i][t]，**用原始 attn**（sink 不置零）。
      4. 分页：零填充到 block_size 的整数倍（nb = ceil(T / block_size) 页），每页取 max。
      5. 选择：每页分数取 top-num_retained_blocks，平手按页号小的在前。

    返回 5 键 dict：
      entropies       : float [B, H_Q]
      group_weights   : float [B, H_KV, r]
      page_scores     : float [B, H_KV, nb]
      retained_pages  : int64 [B, H_KV, nb]，0/1
      retained_tokens : int64 [B, H_KV, T]，0/1（页掩码按 t // block_size 展开）

    非法输入抛 ValueError：
      - `attn` 不是浮点张量、维度不是 3、或含负值；
      - num_retained_blocks / num_query_heads / num_kv_heads / block_size / sink_size
        任一不是 int（bool 也算不合法），num_retained_blocks < 0，sink_size < 0，
        block_size / num_query_heads / num_kv_heads < 1；
      - num_query_heads != H_Q，或 H_Q 不能被 num_kv_heads 整除；
      - num_retained_blocks > nb；
      - tau_g / alpha 不是实数或 <= 0。
    """
    # TODO
    raise NotImplementedError
