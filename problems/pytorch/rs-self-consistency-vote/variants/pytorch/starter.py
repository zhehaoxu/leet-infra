def solve(samples, weights=None):
    """self-consistency（2203.11171）的多数投票账：答案归一化 + 加权计票。

    参数
    ----
    samples : list[str]，每条候选输出的最后一段（调用方已截取答案部分）
    weights : None 或 list[float]，与 samples 等长的票重（缺省全 1）

    归一化规则（题面钉死，依次执行）：
        1. 去首尾空白
        2. 全部转小写
        3. 去千位逗号："12,345" -> "12345"（仅当逗号前后都是数字）
        4. 去尾随句点（"42." -> "42"，仅一个）
        5. 分数统一为 "a/b"："a ÷ b" 与 "a / b"（斜杠两侧空格任意）都写作 "a/b"，
           其中 a、b 是正整数（前后无其他数字）
        6. 百分号剥离："42%" -> "42"
        7. 货币符号剥离：前导 "$" 剥离
        8. 其余字符原样保留（不解析数学等价，"1/2" 与 "0.5" 是不同答案）

    返回
    ----
    dict：
        normalized  list[str]      每条样本的归一化结果（与 samples 等长）
        tally       dict           归一化答案 -> 累计票重（键按首次出现顺序）
        winner      str            票重最大的答案（并列取首次出现者）
        vote_share  float          winner 的票重占总票重的比例（总票重 0 时取 0.0）
        n_candidates int           tally 的键数
    """
    raise NotImplementedError
