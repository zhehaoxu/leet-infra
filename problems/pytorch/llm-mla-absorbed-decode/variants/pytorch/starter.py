def solve(q_nope, q_pe, c_kv, k_pe, W_UK, W_UV, scale):
    """MLA 解码时的「权重吸收」注意力。

    q_nope:(B, H, d_h)  每头的非 RoPE 查询分量
    q_pe:  (B, H, d_r)  每头已施加 RoPE 的查询分量
    c_kv:  (B, S, d_c)  潜在 KV 缓存
    k_pe:  (B, S, d_r)  所有头共享、已施加 RoPE 的键分量
    W_UK:  (H, d_h, d_c) 把潜在向量升维成 k_nope 的每头投影
    W_UV:  (H, d_h, d_c) 把潜在向量升维成 v 的每头投影
    scale: softmax 前的缩放因子

    吸收形式（全程在 d_c 维潜在空间里做注意力）：
      q_abs = q_nope^T W_UK        -> (B, H, d_c)
      s     = (q_abs · c_kv + q_pe · k_pe) * scale
      p     = softmax(s, dim=-1)
      z     = p · c_kv             -> (B, H, d_c)   加权求和后的潜在向量
      out   = W_UV z               -> (B, H, d_h)   每头一次升维

    返回 float32 张量，形状 (B, H, d_h)。
    """
    # TODO
    raise NotImplementedError
