import math


def solve(query_emb, memories, k, recency_decay=0.995, gw=(0.5, 3, 2)):
    """转写 joonspk-research/generative_agents `cognitive_modules/retrieve.py`
    new_retrieve 的三因子检索账（embedding 全显式入参，无 LLM 调用）。

    参数
    ----
    query_emb : [D]          focal point 的 embedding
    memories  : list[dict]，按 last_accessed 时间升序（即时间线顺序）：
                {'emb': [D], 'importance': float}   # importance 即 poignancy
    k         : int，取前 k 条（>0）
    recency_decay : float，上游 recency_decay（缺省 0.995）
    gw        : (w_recency, w_relevance, w_importance)，上游 gw = [0.5, 3, 2]

    记账（题面钉死，与上游一致）：
      recency[i]   = recency_decay ** (i+1)，i 为时间线序号（0 起）
      relevance[i] = cos_sim(mem.emb, query_emb)
      importance[i]= mem.importance
      三个分量各自 min-max 归一化到 [0, 1]；若分量取值范围 range == 0
      （含单节点），全部置 0.5（上游 (target_max-target_min)/2）
      score[i]     = gw[0]*recency_n[i] + gw[1]*relevance_n[i] + gw[2]*importance_n[i]
      排序：按 score 降序（稳定排序，并列保持时间线顺序），取前 k 条

    返回
    ----
    dict：
      scores      list[float]  每条记忆的加权分（原时间线顺序）
      top         list[int]    前 k 条的时间线索引（并列按时间线序）
      components  dict：{'recency': [...], 'relevance': [...], 'importance': [...]}
                  归一化后的三分量（原时间线顺序）
    """
    raise NotImplementedError
