import torch


def solve(query, key, value, wo, window_size=32, max_capacity_prompt=4096,
          kernel_size=5, pooling="avgpool", metric_mode="after", use_wo=True,
          use_norm=False, use_ema=True, alpha=0.3, tau=1.0):
    """ReST-KV 的重要性打分：赔率项 x 输出重建乘子，再空间平滑后 top-k。

    query / key / value : 同形状同 dtype 的 float32/float64 (B, H, S, D)；K/V 已按
        query head 展开，本题不做 GQA 复制
    wo                  : 同 dtype 的 (C_out, H * D) 输出投影权重，use_wo=False 时不
        参与计算但仍校验
    记 L = S - window_size（历史段长度）、k = max_capacity_prompt - window_size（名额）

    1. `A = query[..., -W:, :] @ key.transpose(2,3) / sqrt(D)`，给右下角 W x W 块
       **加**上严格上三角的 `finfo(A.dtype).min`（窗内第 i 行只看前 i+1 个 key），再
       `softmax(dim=-1, dtype=float32)` 转回原 dtype；被掩位置概率恰为 0。
    2. `A_past = A[:, :, -W:, :-W]`，形状 (B, H, W, L)。
    3. `metric_mode == "after"`：`p_bar` = A_past 沿 query 轴的均值，或 use_ema 时的
       EMA（`e_0 = A_past[0]`、`e_i = alpha*A_past[i] + (1-alpha)*e_{i-1}`，只回吐
       最后一个状态）；`v_bar = einsum("bhl,bhld->bhd", p_bar, value[:, :, :-W])`；
       `score_l = p_bar_l / (1 - p_bar_l)` 再乘 `||W_o v_bar - W_o V_l||_2 ** tau`
       （use_wo）或 `||v_bar - V_l||_2`（不开 use_wo，这一支没有 tau）。
    4. `metric_mode == "before"`：`O = A @ value`（整段 softmax 乘**全量** value）；
       `u = A / (1 - A)` 形状 (B, H, W, S)；乘子逐 query 逐位置（use_wo 时在输出空间
       取差，否则在 value 空间）；use_norm 时再除以 `||W_o O_q||_2`（不开 Wo 时除
       `||O_q||_2`）；**然后**才切 `[-W:, :-W]` 并沿 query 轴均值或 EMA。
       use_norm 与 tau 只在 before 支有意义，after 支根本不读 use_norm。
    5. 平滑：avgpool / maxpool 用 kernel_size（奇数）、padding=kernel_size//2、
       stride=1 的一维池化，(B, H, L) 按 (batch, channel, length) 逐 head 独立；
       none 原样。
    6. 选择：仅当 `S >= max_capacity_prompt` 时取 pooled 的 top-k（分数降序，同分取
       下标小的）；kept_positions = 这批位置 + 末窗 L..S-1 升序。S 不够时不选择，
       但第 1-5 步的分数照常算并回吐。

    返回 5 键 dict：
      attn_weights     : (B, H, W, S)，与输入同 dtype
      important_score  : (B, H, L)，平滑前
      pooled_score     : (B, H, L)，平滑后
      selected_indices : int64 (B, H, k)，历史段下标（0 起即绝对下标），选择序
      kept_positions   : int64 (B, H, k + W)，前 k 个跟着选择序、末窗升序

    非法输入抛 ValueError：query/key/value 不是四维、形状或 dtype 不一致、不是
    float32/float64；wo 不是二维、dtype 不一致或 shape[1] != H*D；S <= W、W < 1、
    max_capacity_prompt <= W、kernel_size 为偶数或小于 1；pooling / metric_mode 取
    未知字符串；use_wo / use_norm / use_ema 不是布尔；alpha 不在 [0,1]、tau < 0；
    整数参数不是 int（bool 也算不合法）。
    """
    # TODO
    raise NotImplementedError
