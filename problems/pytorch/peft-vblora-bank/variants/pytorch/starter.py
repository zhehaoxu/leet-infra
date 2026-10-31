def solve(x, W0, logits_A, logits_B, vector_bank, b0=None, topk=2):
    """VB-LoRA：向量库选择的极低秩适配器（对齐 peft `tuners/vblora/layer.py` 的
    `_get_lora_matrices` + `get_delta_weight` + `forward`）。

    形状约定
        x            : (..., d_in)，任意前导维度
        W0           : (d_out, d_in) 基座权重
        b0           : (d_out,) 或 None
        vector_bank  : (num_vectors, vector_length)，全局共享的一张向量库
        logits_A     : (r, d_in // vector_length, num_vectors)
        logits_B     : (d_out // vector_length, r, num_vectors)
        topk         : 正整数，且 1 <= topk <= num_vectors

    逐 logit 向量的选择（对最后一维长度为 num_vectors 的每个向量独立执行）
        1) 在 num_vectors 个 logit 上取最大的 topk 个；同值时按下标升序取
           （等价的稳定选择口径，不依赖任何库的默认并列行为）。
        2) softmax **只对选中的这 topk 个 logit** 归一化（不是对整条 num_vectors）。
        3) 用这 topk 个权重去加权 vector_bank 里对应的 topk 行，求和得到一个
           vector_length 维向量。

    低秩矩阵的拼装
        A (r, d_in)：把 logits_A 的选择结果 (r, d_in//vl, vl) 按 tile 顺序展平，
                     即 A[:, t*vl:(t+1)*vl] 来自 logits_A 的第 t 个 tile。
        B (d_out, r)：把 logits_B 的选择结果 (d_out//vl, r, vl) 先转置成
                     (out_tile, vl, r) 再展平前两维，即
                     B[o*vl + s, j] = 第 o 个输出 tile、第 j 条秩、第 s 个分量。
        delta = B @ A（没有 alpha/r 之类的额外缩放），形状 (d_out, d_in)。

    前向
        y = x @ W0.T + b0 + x @ delta.T
      （b0 为 None 时不加偏置；等价于 base(x) 再加上把 x 先乘 A 再乘 B 的结果。）

    返回 (y, A, B, delta)：y 与 x 的前导维度一致、最后一维为 d_out；
    A 为 (r, d_in)；B 为 (d_out, r)；delta 为 (d_out, d_in)。
    内部全部用 float64 计算，返回的四个张量都是 float32。
    非法输入（形状不匹配、整除关系不成立、topk 越界、非浮点、NaN/Inf 等）
    抛出 ValueError。禁止调用现成 PEFT 库。"""
    raise NotImplementedError
