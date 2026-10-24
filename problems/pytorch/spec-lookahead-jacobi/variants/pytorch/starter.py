import torch


def solve(prompt, steps, min_n, max_n, pool_seed, V):
    """Lookahead 解码（Jacobi n-gram 池）：pool{n-gram->下一 token}，由 pool_seed 各序列的
    所有长度∈[min_n,max_n] 子串→后继 预置。每步：用 seq 的最长后缀 n-gram（max_n..min_n）查草稿，
    命中则取之，否则确定性回退 (seq[-1]+1)%V；追加后再把「紧邻新 token 的前缀 n-gram→新 token」写回池。
    返回 torch.long 一维、长度 steps 的生成序列。"""
    # TODO
    raise NotImplementedError
