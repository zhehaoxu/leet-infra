import torch


def solve(p, qs):
    """多 draft 级联的残差逐级补齐（同一位置的 L 个 draft 依次填坑）。
    p:(T,) float32 target 分布；qs:(L,T) float32，第 l 行是第 l 级 draft 的分布。
    逐级维护未解释质量 rem（初始为 p）：本级能贡献的是 acc_l = min(qs[l], rem)，
    acc_l 计入「命中质量」、同时从 rem 里扣掉；L 级都走完后剩下的 |rem| 是退回 target
    重采的概率，out = Σ acc_l + |rem|·p。
    返回 (out:(T,) float32、accepted_mass:(L,) float32 每级贡献的质量、
    target_mass:float32 标量 = |rem| 退回 target 的质量)。"""
    # TODO
    raise NotImplementedError
