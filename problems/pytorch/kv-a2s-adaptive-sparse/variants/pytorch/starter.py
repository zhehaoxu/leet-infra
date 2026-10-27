import torch


def solve(attn, k, p):
    """A2S：用上一步的 top-k 稀疏模式当作候选集，并按当步累计质量自适应定 n。
    attn:(S,L) float32 每步注意力行，第 s 行只有前 P+s 列有效（P=L-S）；
    k 为候选集大小，p∈[0,1] 为目标质量占比。
    第 0 步：候选 = 全部有效列，按其自身分数降序（并列取小下标）；
    第 s>0 步：候选 = 第 s-1 步有效列上的 top-k，再按**当步**分数降序；
    n = 最小的 m（1..|候选|）使候选前 m 名的当步分数累计 >= p*当步总质量，否则取满。
    返回 (n_adaptive:(S,) int64, keep:(S,L) bool)。
    k<1 或 p 不在 [0,1] 抛 ValueError。"""
    # TODO
    raise NotImplementedError
