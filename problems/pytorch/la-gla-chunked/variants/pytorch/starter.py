import torch


def solve(q, k, v, g, chunk_size, h0=None, scale=None):
    """转写 fla `naive_recurrent_gla` 递推的 chunk 并行形式（log-space 分块恒等式）。

    朴素递推（官方 naive_recurrent_gla，每头单序列，本题无 batch/head 维）：
        h = h0 或 0                        # [K, V]
        for t in range(T):
            h = h * exp(g[t]) + k[t] ⊗ v[t]   # g 逐 key 通道 log 衰减
            o[t] = (q[t] * scale) @ h
        scale 缺省 = K ** -0.5

    chunk 并行形式（把 [0, T) 切成左闭右开的 chunk_size 段；每段先读段前状态）：
        设段内含入 cumsum：S_t = sum_{j<=t, j∈chunk} g_j（段内相对量）
        输出：  o_t = (q_t ⊙ exp(S_t))·scale @ h_prev            # 段间贡献
                     + sum_{j<=t, j∈chunk} (q_t ⊙ exp(S_t - S_j))·scale · k_j · v_j
                                                              # 段内贡献
        状态：  h_next = exp(S_end)·h_prev
                       + (k ⊙ exp(S_end - S))ᵀ @ v              # S_end = 段末 S

    参数
    ----
    q : [T, K]   k : [T, K]   v : [T, V]   g : [T, K]（log 衰减，通常 ≤ 0）
    chunk_size : 正整数
    h0   : None 或 [K, V]
    scale: None 则取 K ** -0.5

    返回
    ----
    dict：
        o            [T, V]
        final_state  [K, V]
        chunk_states list[[K, V]]  每个 chunk 更新完的状态（账本）

    全程 float64 计算（题面钉死，保证与朴素递推逐位对拍）。
    """
    raise NotImplementedError
