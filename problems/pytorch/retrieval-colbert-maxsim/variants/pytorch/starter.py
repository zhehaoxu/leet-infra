import torch


def solve(query, doc, chunk_size):
    """ColBERT MaxSim 打分 + MaxPooled 长文档聚合（CPU、确定性）。

    参数：
    - query: (Lq, D) 浮点张量，query 词向量；逐行做 L2 归一化后使用。
    - doc: (Ld, D) 与 query 同 dtype 的浮点张量，文档词向量；同样逐行
      L2 归一化。零向量（范数为 0）归一化后保持全 0，它与任何向量的
      余弦相似度按 0 处理，不产生 NaN。
    - chunk_size: int >= 1。把文档按 token 顺序硬切成块：第 c 块覆盖
      [c*chunk_size, min((c+1)*chunk_size, Ld))，尾块不足 chunk_size
      也保留；块数 C = ceil(Ld / chunk_size)，Ld = 0 时 C = 0。

    返回五元组：
    - base_score: 0 维浮点张量，S = sum_i max_j cos(q_i, d_j)。Ld = 0
      时每个 max 按 0 处理，S = 0；Lq = 0 时也为 0。
    - base_idx: (Lq,) int64，base_idx[i] 是使 cos(q_i, d_j) 最大的最小
      文档下标 j（并列取小下标）；Ld = 0 时全为 -1。
    - chunk_maxsim: (Lq, C) 与输入同 dtype，chunk_maxsim[i, c] =
      max_{j 属于第 c 块} cos(q_i, d_j)。
    - maxpooled_score: 0 维浮点张量，sum_i max_c chunk_maxsim[i, c]，
      即同一 query 词的贡献只计一次；由定义它恒等于 base_score。
    - repeated_score: 0 维浮点张量，sum_{i,c} chunk_maxsim[i, c]，即
      「每个 chunk 各算一次再相加」的重复计分口径，仅用于对照；单
      chunk 时等于 base_score，多 chunk 时一般更大（余弦非负时必不
      小于 base_score；若所有块内最大值都为负则也可能更小）。

    错误约定：query / doc 非二维、特征维不一致、dtype 不一致或非浮点、
    chunk_size < 1 均抛 ValueError。查询为空（Lq = 0）或文档为空
    （Ld = 0）都合法，不抛错。
    """
    # TODO
    raise NotImplementedError
