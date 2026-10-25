import torch


def solve(hypotheses, references):
    """ROUGE-1 / ROUGE-2 / ROUGE-L。hypotheses: list[list[int]]（token id 序列，已切好，不用分词器）；
    references: list[list[list[int]]]，references[i] 是第 i 条假设的参考列表（≥1 条，可多条）。

    对每阶指标：先对每条假设的**每一条参考**单独算 F1（ROUGE-n 用按参考频次裁剪的 n-gram 重合，
    ROUGE-L 用 LCS 长度；两者 R=P=0 时 F1 记 0），**取其中的最大值**作为该假设的得分，再对全部假设
    求算术平均（空假设也是一条假设，贡献 0）。

    返回三元组 (rouge1_f1, rouge2_f1, rouge_l_f1)，都是 Python float。禁止使用 rouge_score / nltk。"""
    # TODO
    raise NotImplementedError
