import torch


def solve(scores, rels, k, exp_gain=True):
    """nDCG@k。scores: list[float]，ranking 模型给候选文档的打分，列表下标就是文档 id；
    rels: list[int]，与 scores 一一对应的分级相关性标注（非负整数，越大越相关）；
    k: int，截断位置；exp_gain: bool，True 用增益 2^rel - 1，False 用线性增益 rel。

    排序规则：按 score **降序**；分数相同的文档按**文档下标升序**（稳定 tie-break）。
    取前 min(k, N) 个位置算 DCG，折损为 1 / log2(i + 1)，i 从 1 开始（第 1 位折损是 1）。
    IDCG 把 rels 按降序重排（理想排序）后按同样公式算。
    IDCG == 0（全标注为 0 或无候选）⇒ nDCG 记哨兵 0.0。

    返回三元组 (ndcg, dcg, idcg)，都是 Python float。禁止 sklearn（判题环境里也没有）。"""
    # TODO
    raise NotImplementedError
