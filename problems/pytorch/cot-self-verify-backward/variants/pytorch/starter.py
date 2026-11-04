def solve(question, sampled_answers, verifier_outputs, ground_truth,
          declarative=""):
    """Self-Verification（2212.09561, WENGSYX/Self-Verification）后向验证账。

    参数
    ----
    question : str
        问题原文（含末尾句号；官方 'Answer Choices' 前缀会先被切掉语义
        不入本题——数值路径直接消费全文）。
    sampled_answers : list[str]
        k 份 CoT 采样经 answer_cleansing 后的答案串（GPT 与清洗不入判题
        域）；solve 内做 float 规范化 + 非空首现去重得候选集 answers。
    verifier_outputs : list[list[str]]
        扁平判定串池：第 `a_idx*3 + d` 个元素是「候选 a_idx 的第 d 个
        挖空子问题」的 verifier 判定串列表（官方每子问题采样 10 次）；
        缺失按空列表（0 分）处理。
    ground_truth : str
        金标答案（与候选同样做 float 规范化后比较）。
    declarative : str，默认 ""
        上游改写句缓存：非空且含 answers[0] 时把最后一次出现的
        answers[0] 替换为当前候选（官方反向定位拼接 ≡ rfind 替换）；
        为空或不含 answers[0] 时用 fallback 模板
        "{question} The answer is {answer}."（GPT 重生成不入判题域）。

    账本规则（官方逐条钉死）：
        1. 候选收集：str(float(x)) 规范化（"5"→"5.0"，非数值原样）、
           非空、首现去重；
        2. 数字挖空：text 逗号先转句点，在最后一个句号处截断 ts，
           findall(r'-?\\d+\\.?\\d*') 抽全部数字，对每个数字把 ts.split('.')
           中**每个**含它的句段的数字区间（字符列表切片赋值）换成 "'X'"
           （官方 number_find 置位缺失——多句段全换是忠实行为），
           重组后 replace('..','.')、去首空格/补尾空格/replace(' .','.')，
           子问题 = '"' + new_ts + declarative + '"'，**最多取前 3 个数字**；
        3. verifier 评分：判定串逐个 try float；期望值 try float(挖空
           数字)；score_d = sum(v == expected)，候选总分 = Σ_d score_d；
        4. 单候选：scores={0:1}，不构造子问题不评分；
        5. 选择：argmax(scores) **平局取最前**（np.argmax 语义）；
           is_true = 各候选规范化后是否等于规范化 GT。

    返回
    ----
    dict：
        answers        list[str]    规范化去重后的候选（首现序）
        sentences      list[list[str]]  每候选的挖空子问题串（≤3）
        sub_numbers    list[list[str]]  每候选被挖空的数字（≤3）
        scores         dict         候选下标 -> verifier 总分
        chosen_idx     int          胜出候选下标
        chosen_answer  str          胜出答案（规范化后）
        is_true        list[bool]   各候选是否命中 GT
        chosen_correct bool         胜出答案是否命中 GT
    """
    raise NotImplementedError
