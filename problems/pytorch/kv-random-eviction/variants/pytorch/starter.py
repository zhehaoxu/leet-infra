import torch


def solve(draws, attn_weights, n_sink, window_size, compression_ratio):
    """Random eviction 基线：强制保留 F=[0,min(n_sink,L))∪[L-min(window_size,L),L)；
    n_kept=max(1,int(L*(1-cr)))，名额 q=max(0,n_kept-|F|)；中段按 draws 降序取 q（平局小位置优先，
    不足全取）；kept=F∪选中 升序 int64；coverage=Σattn[kept]/Σattn（Σ=0 时用 |kept|/L）。
    返回 (kept, coverage float)。draws∈[0,1) (L,)，attn_weights 非负 (L,)，CPU float32。"""
    # TODO
    raise NotImplementedError
