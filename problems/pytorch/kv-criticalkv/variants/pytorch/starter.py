import torch


def solve(scores, wov_norm, compression_ratio, first_stage_ratio, epsilon):
    """CriticalKV 两阶段：n_kept=max(1,int(L*(1-cr))); b1=int(L*(1-cr)*fsr)（fsr>0 时 max(1,b1)）;
    阶段1取 base 的 top-b1（平局小位置优先），final=(base+eps)*wov_norm 后在阶段1位置 scatter float32 上限，
    再取 final 的 top-n_kept。返回 (kept int64 (H,n_kept) 升序, final float32 (H,L))。输入 CPU float32, wov_norm>=0。"""
    # TODO
    raise NotImplementedError
