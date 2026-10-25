import torch


def solve(last_attn, n_initial, n_recent, n_last, lazy_threshold):
    """SimLayerKV：逐层 score=Σ last_attn[:, :n_initial] + Σ last_attn[:, -n_recent:]（按 L 截断），
    score 严格 > lazy_threshold 才懒惰；懒惰层只留 [0,n_initial)∪[L-(n_recent-n_last),L)，
    ratio=(L-n_initial-(n_recent-n_last))/L，否则全留 ratio=0。
    lazy_threshold==1.0 或 L<=n_initial+n_recent+n_last 时整层不压缩。
    返回 (mask bool (N,L), ratios float32 (N,))。last_attn (N,L) 非负。"""
    # TODO
    raise NotImplementedError
