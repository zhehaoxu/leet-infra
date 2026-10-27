import torch


def solve(alphas, cost_ratio):
    """接受率 → 期望吐词数 / 加速比的解析账，并反解最优 γ。
    alphas:(G,) float32，第 i 个是 draft 第 i+1 个位置的逐位接受率（γ 上界为 G）；
    cost_ratio 为「draft 每 token 时延 / target 单步时延」。
    对 γ ∈ [0, G]：期望吐词 e(γ) = 1 + Σ_{i=1..γ} Π_{j=1..i} α_j，
    加速比 s(γ) = e(γ) / (1 + γ·cost_ratio)。返回 (expected:(G+1,)、speedup:(G+1,)、
    best_gamma:int64 标量（并列取最小 γ）、best_speedup:float32 标量)。"""
    # TODO
    raise NotImplementedError
