import torch


def solve(x, W0, b0, adapters):
    """GLoRA 合并前向（对齐 peft tuners/glora.py 的 merge + forward）。
    x:(...,d_in)  W0:(d_out,d_in)  b0:(d_out,) 或 None。
    adapters: 非空 list[dict]，每项一个适配器，字段：
      "A_d":(d_out,r) "A_u":(r,d_in) -> A = A_d @ A_u
      "B_d":(d_out,r) "B_u":(r,d_in) -> B = B_d @ B_u
      "C":(d_in,) "D":(d_out,) "E":(d_out,)（仅 b0 不为 None 时需要）。
    合并：W' = W0 + Σ_i (W0 ⊙ A_i + B_i)，每一项都以原始 W0 计算；
          b' = b0 + Σ_i (b0 ⊙ D_i + E_i + W0 @ C_i)。
    返回 (y, W', b')，y = x @ W'.T + b'；b0 为 None 时 b' 也为 None。
    内部 float64、返回 float32。禁止调用现成 PEFT 库。"""
    # TODO
    raise NotImplementedError
