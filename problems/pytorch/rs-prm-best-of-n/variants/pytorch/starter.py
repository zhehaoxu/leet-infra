def solve(step_scores, answer_scores=None, rule="product", top_m=1):
    """PRM（2305.20050 "Let's Verify Step by Step"）best-of-N 的步级聚合账。

    参数
    ----
    step_scores : list[list[float]]，N 条候选各自的步级打分序列（长度可不同；
                  每条至少 1 步）
    answer_scores : None 或 list[float]，与 step_scores 等长的终答打分。
                  None 时聚合只用步级分；提供时聚合结果 = 步级聚合 × answer_score。
    rule : "last" | "max" | "min" | "product"（四种归并口径，与论文口径同名）
        last    取最后一步
        max     取最大步分
        min     取最小步分
        product 全步连乘
    top_m : int >= 1，返回票选出的前 m 条候选 index（按聚合分降序，
            并列取 index 较小者）

    返回
    ----
    dict：
        agg        list[float]  每条候选的聚合分（answer_scores 给出时含乘项）
        best       int          agg 最大的候选 index（并列取较小）
        top        list[int]    前 top_m 个 index（降序，并列取较小）
        n_steps    list[int]    每条候选的步数
        min_len    int          最短候选的步数
        tie        bool         最高分是否并列（严格计数：agg.count(max) > 1）

    校验：rule 不在四种之列、top_m < 1、step_scores 为空或某条无步、
    answer_scores 长度不匹配 -> ValueError。
    """
    raise NotImplementedError
