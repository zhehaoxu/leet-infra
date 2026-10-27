import torch


def solve(A, B, gA, gB, lr, lam, alpha, r):
    """一步 SGD 后的 LoRA 更新量，两种缩放：rsLoRA α/√r 与普通 α/r。
    A:(r,d_in) B:(d_out,r) 梯度同形；A'=A-lr*gA，B'=B-lam*lr*gB；
    返回 (delta_rslora, delta_vanilla)，均 (d_out,d_in) float32。禁止现成 PEFT 库。"""
    # TODO
    raise NotImplementedError
