def solve(word: str, merges: list) -> list:
    """用有序 BPE merges 把词切成子词。

    word   : 小写字母串
    merges : [(a, b), ...]，下标 = 优先级 rank（越小越先应用）

    算法：符号序列 = list(word) + ['</w>']；反复在当前相邻对里取「命中 merges 且 rank 最小」
    的一对，不重叠、从左到右替换成拼接符号 a+b，直到无可用对。返回子词 list[str]。
    """
    # TODO: 实现 BPE 编码
    raise NotImplementedError
