import torch


def solve(hypotheses, references, max_n=4, smoothing=False):
    """语料级 BLEU。hypotheses: list[list[int]]（每条是一串 token id，已是分好词的 id 序列）。
    references: list[list[list[int]]]，references[i] 是第 i 条假设对应的**参考列表**（≥1 条，
    通常 1~4 条）。max_n 控制 1..max_n 阶 n-gram，smoothing 为真时对每阶精度做加一平滑。

    返回 (bleu, precisions)：bleu 是 Python float，precisions 是 float32 张量 (max_n,)，
    为**真正进入几何平均**的各阶裁剪精度 p_1..p_max_n。禁止调用任何 BLEU 实现库。"""
    # TODO
    raise NotImplementedError
