import torch


def solve(K, V, bits):
    """KIVI 非对称 KV 量化：Key 逐通道（dim=0 统计）、Value 逐 token（dim=1 统计）。
    K,V:(T,D) float32；bits 在 [2,8]。
    Δ=(max-min)/(2^bits-1)（退化时 Δ=1）；q=clamp(round((x-min)/Δ),0,2^bits-1)；
    deq=q*Δ+min。返回 (K_deq,V_deq,K_scale,K_zero,V_scale,V_zero)，
    反量化 float32 (T,D)，参数 float32（K 长 D、V 长 T）。
    bits 非法抛 ValueError。"""
    # TODO
    raise NotImplementedError
