import torch


def solve(text, pattern):
    """在线构建后缀自动机（SAM），再沿转移边走最长后缀匹配。
    text:(N,) int64 语料（N>=1）、pattern:(P,) int64 查询（可为空）。
    返回 (n_states: int64 标量（含初始状态）、distinct: int64 标量（text 的不同子串数，
    = Σ_{s≠root} len[s] − len[link[s]]）、match_len:(P,) int64，第 i 项是 pattern 的前
    i+1 个 token 里「最长的、同时是 text 子串」的后缀长度）。"""
    # TODO
    raise NotImplementedError
