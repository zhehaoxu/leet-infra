import torch


def solve(alphas, budget):
    """一个 target forward 的固定名额怎么分给批内各请求，使期望吐词数最大。

    alphas:(N,) float32，第 i 个请求每多草拟一位的接受率 α_i（各位置同分布）;
    budget:int，本步 target 侧可用的验证名额总数。

    若给请求 i 分配 γ_i 位草稿，它占用 γ_i + 1 个名额（草稿位 + 自己那一列验证），
    期望吐词 e_i(γ) = 1 + α_i + α_i^2 + ... + α_i^γ。
    求在 Σ(γ_i + 1) <= budget 下最大化 Σ e_i(γ_i)，返回
    (gammas:(N,) int64、expected:(N,) float32、total_expected:float32 标量)。

    名额不够每个请求至少 1 个、或者 alphas 非法时抛 ValueError。
    """
    # TODO
    raise NotImplementedError
