def solve(samples_per_question, sort_by="disagreement", dataset_kind="generic",
          qes_limit=0):
    """Active-Prompt（2302.12246, shizhediao/active-prompt）不确定度选样账。

    参数
    ----
    samples_per_question : list[list[str]]，每题 k 个采样答案
        官方 GPT k 次采样 + answer_extraction 之后的**抽取后答案串**
        （判题域不含采样与抽取）；空串 "" 表示该次采样无解。
    sort_by : "disagreement"（默认）| "variance" | "entropy"
        官方 --sort_by 三选一。
    dataset_kind : "generic"（默认）| "numerical" | "strategyqa"
        官方数据集族映射：gsm8k/svamp/asdiv/addsub/singleeq/multiarith →
        numerical（有 variance 且 occurrence 无预置键）；strategyqa →
        occurrence 预置 {"yes":0,"no":0} 且 disagreement 排序走特例；
        其余 → generic。
    qes_limit : int，默认 0
        只计算前 qes_limit 题（0 = 全部，超过题数即全部）；
        排序只作用于被计算的子集。

    账本规则（官方 uncertainty_estimation.py 逐条钉死）：
        1. occurrence 计数：非空答案按原串为键（首现序），空答案计入
           哨兵键 NO_SOLUTION = '-10086'；
        2. variance 仅 numerical：occurrence 展开成 float 列表
           （NO_SOLUTION 也以 -10086 计入），总体方差 ddof=0；
        3. entropy 走 scipy.stats.entropy 口径：
           -Σ_{c>0} (c/n)·ln(c/n)，0 频率项（yes/no 预置零频）贡献 0，
           频率全零为 nan（判题环境无 scipy，自实现对齐）；
        4. disagreement = len(occurrence)（含 NO_SOLUTION）；
        5. 排序（stable，平局保持输入序）：
           disagreement → -len(occurrence) 降序；strategyqa 特例 →
           abs(yes-no) **升序**（两票越接近越不确定；官方 except 兜底
           属死分支）；variance → -variance 降序且**仅 numerical**
           （官方 elif 带 dataset 过滤，不匹配静默不排序）；entropy →
           -entropy 降序。

    返回
    ----
    list[dict]（排序后）：每条
        dataset_idx   int     原输入下标
        variance      float   仅 numerical 键
        entropy       float
        occurrence    dict    答案串 -> 次数（首现序）
        disagreement  int     distinct 答案数（含 NO_SOLUTION）
    """
    raise NotImplementedError
