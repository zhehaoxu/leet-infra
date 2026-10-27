import torch


def solve(E, A, B, g1_E, g1_A, g1_B, g2_E, g2_A, g2_B, beta1, beta2, budget):
    """AdaLoRA 一次预算重分配：两步 EMA 敏感度评分 + 全局第 k 小截断。
    E:(r,1) A:(r,d_in) B:(d_out,r)，梯度同形；返回 (scores:(r,), masked_E:(r,1))，float32。
    掩码只置零 E 中 score <= 第 k 小（k=r-budget，k<1 不掩码）的位置。禁止现成 PEFT 库。"""
    # TODO
    raise NotImplementedError
