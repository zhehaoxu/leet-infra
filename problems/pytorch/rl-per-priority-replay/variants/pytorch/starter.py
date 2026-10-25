import torch


def solve(priorities, uniform_draws, alpha, beta):
    """优先经验回放（PER, Schaul et al. arXiv:1511.05952）的采样与重要性采样权重。
    随机性不作弊：均匀随机数由入参显式给出。

    入参：
      priorities   (N,) float，回放池每条样本的优先级 p_i >= 0
      uniform_draws(K,) float，已抽好的均匀随机数 u_k ∈ [0, 1)
      alpha        标量 >= 0，优先级指数
      beta         标量 >= 0，重要性采样修正指数
    返回三元组 (indices, weights, probs)：
      probs    (N,) float32 采样概率 P(i) = p_i**alpha / Σ_j p_j**alpha
      indices  (K,) int64   每条均匀随机数落到的样本下标：
               最小的 i 使 cumsum(P)[:i+1] > u_k（浮点越界时截到 N-1）
      weights  (K,) float32 IS 权重 w_k = (N*P(indices[k]))**(-beta)，
               再除以本批 max_j 归一（故 max 恒为 1）
    退化：alpha=0 时 P≡1/N；beta=0 或 N*P≡常数 时 w≡1；
      Σ_j p_j**alpha == 0（如全零优先级且 alpha>0）时视为均匀分布 P=1/N；
      N==0（空池）时 indices 全 -1、weights 全 0、probs 为空。
    K=0 返回空 indices/weights 与完整 probs。"""
    # TODO
    raise NotImplementedError
