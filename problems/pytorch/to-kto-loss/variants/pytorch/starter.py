import torch


def solve(log_ratios, labels, z_ref, beta, lambda_D, lambda_U):
    """KTO 损失：desirable(label=1) λ_D*(1-σ(β(z-z_ref)))；undesirable(label=0) λ_U*(1-σ(β(z_ref-z)))。
    log_ratios,labels 形状 (N,)；labels 取 0/1。返回 0 维标量=均值。禁止现成对齐库。"""
    # TODO
    raise NotImplementedError
