def solve(questions, rationales, pred_anses, cluster_assignment, dist,
          max_ra_len=5, arithmetic=True, gold_anses=None):
    """Auto-CoT（2210.03493, amazon-science/auto-cot）聚类选代表拼 demo。

    参数
    ----
    questions / rationales / pred_anses : 等长 list[str]
        候选池：问题原文、zero-shot-CoT 推理链、模型预测答案
        （官方 log 解析层 Q:/A:/pred_mode 的产物直接作入参）。
    cluster_assignment : list[int]，与样本等长
        每条样本的簇编号（官方 KMeans.labels_）。
    dist : list[list[float]]，形状 (n, k)
        每条样本到各簇质心的欧氏距离（官方 clustering_model.transform）。
    max_ra_len : int，默认 5
        rationale 最大行数上限（官方 last_letters 任务传 7）。
    arithmetic : bool，默认 True
        是否启用算术任务的答案回验（gsm8k/multiarith/singleeq/addsub/
        svamp 一族：pred_ans 必须出现在 rationale 倒数第二句或最后 10 词）。
    gold_anses : None 或 list[str]
        官方 debug=True 才收集的金标答案，给出时 strip 透传。

    管线（官方 run_demo.py sampling=center 路径）：
        1. 分簇：第 i 簇收集 assignment==i 的样本（按下标升序）；
        2. 簇内按到质心距离**稳定升序**（平局保持全局下标序）；
        3. 逐候选过五重过滤，任一失败换下一候选：
           a. 问题词数 len(question.strip().split()) <= 60
           b. rationale 行数 <= max_ra_len
           c. rationale 以 "." 结尾（空串不通过）
           d. pred_ans strip 后非空
           e. arithmetic 时答案回验（倒数第二句 r.split(".")[-2]
              或尾 10 词 r.split()[-10:]），不过则跳过
        4. 通过者收为该簇 demo 并 break（每簇至多一个）；
        5. rationale 清洗：\\n\\n→\\n→空格、strip、空白折叠单空格；
           question/pred_ans 原样；
        6. 簇按编号 0..k-1 依次处理，全被过滤（或空簇）则该簇无 demo。

    返回
    ----
    dict：
        demos         list[dict]  {question, rationale, pred_ans, gold_ans}
        selected      list[int]   被选中样本的全局下标（按簇序）
        cluster_order list[int]   每个 demo 来自的簇编号（与 selected 等长）
        num_clusters  int         簇数 k
    """
    raise NotImplementedError
