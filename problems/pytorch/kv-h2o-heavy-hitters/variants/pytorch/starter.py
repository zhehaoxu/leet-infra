import torch


def solve(attn, budget, n_sink):
    """H2O：跨步累计注意力 + 每步重击手池。
    attn:(S,L) float32，第 s 行只有前 P+s 列有效（P=L-S 为 prompt 长度），其余为 0；
    每步把该行累加到 acc 上（**不除**已见步数），保留集 = 前 n_sink 个 sink ∪
    acc_s 在有效列上的 top-budget（并列取小下标）。
    返回 (acc:(L,) float32 全程累计, keep:(S,L) bool 每步保留掩码)。
    budget<0 或 n_sink 不在 [0,L] 内抛 ValueError。
    禁止 torch.topk 以外部排序结果代替并列规则（并列必须取小下标）。"""
    # TODO
    raise NotImplementedError
