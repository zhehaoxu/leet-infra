def solve(word_freqs, num_merges: int) -> list:
    """从零训练 BPE：贪心学出前 num_merges 条「最频繁相邻符号对」合并规则。

    word_freqs : [(word:str, freq:int), ...]
    num_merges : 要学习的规则数 K

    算法：每词拆字符 + 追加 '</w>'；反复 (1) 词频加权统计相邻二元组
    (2) 取计数最大、并列取字典序最小的一对 (3) 所有词内不重叠地把它替换成拼接符号。
    按学习顺序返回 [(first, second), ...]（长度 ≤ K）。
    """
    # TODO: 实现 BPE 训练
    raise NotImplementedError
