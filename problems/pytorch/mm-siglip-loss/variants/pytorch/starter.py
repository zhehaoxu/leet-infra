import torch


def solve(sim, logit_scale, logit_bias):
    """SigLIP：z=logit_scale*sim+logit_bias; y=2*eye-1; loss=-(1/B)*Σ log σ(y⊙z)（分母 B、非 B²）。
    sim 为 (B,B) 方阵(对角正对)。返回 0 维标量。logsigmoid 需数值稳定，禁止对比库。"""
    # TODO
    raise NotImplementedError
