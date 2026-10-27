import torch


def solve(cand_tokens, cand_lps, budget, prefix_len):
    """由各头候选构造 draft 树（best-first 扩张），并生成 tree attention 结构。
    cand_tokens:(H,K) int64、cand_lps:(H,K) float32（<=0）。第 h 行给出第 h 个 draft 位置
    的 K 个候选 token 及其 logprob，树深最多 H。按「累积 logprob 降序 + 路径字典序」扩张，
    每吐出一个节点占 1 个 budget，只把已弹出节点的子节点入队（因此节点集前缀封闭）。
    返回 (parent:(N,) int64 父节点下标/-1、node_tokens:(N,) int64、depth:(N,) int64、
    positions:(N,) int64 = prefix_len+depth-1、mask:(N, prefix_len+N) bool)。"""
    # TODO
    raise NotImplementedError
