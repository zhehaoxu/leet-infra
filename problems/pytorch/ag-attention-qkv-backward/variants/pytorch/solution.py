"""参考答案：概率与上游梯度各自成对出现，折头用 einsum 一次收干。

关键三步：
1. dP = dO @ V_headᵀ 把输出梯度翻译回概率空间；
2. softmax 反向 dS = P ∘ (dP − Σ P·dP) · scale，掩码位因为 P=0 自动为 0；
3. 折头：把 (Hq) 拆成 (Hkv, groups) 后用一次 einsum 同时对 groups 与序列轴求和，
   得到 dK/dV —— 这一步写成 for h 累加会慢，写成不折叠的 (B,Hq,·) 则形状必错。
"""

import torch


def solve(Q, K, V, P, dO, scale=1.0):
    if Q.dim() != 4 or K.dim() != 4 or V.dim() != 4:
        raise ValueError("Q/K/V 必须是 (B, H, S, D) 四维张量")
    b, hq, sq, d = Q.shape
    kb, hkv, sk, dk = K.shape
    if V.shape != K.shape:
        raise ValueError("V 必须与 K 同形")
    if kb != b or dk != d:
        raise ValueError("K 的 batch/head_dim 必须与 Q 一致")
    if hq % hkv != 0:
        raise ValueError("Q 的头数必须能被 K/V 的头数整除（GQA）")
    if tuple(P.shape) != (b, hq, sq, sk):
        raise ValueError("P 的形状必须是 (B, Hq, Sq, Sk)")
    if tuple(dO.shape) != (b, hq, sq, d):
        raise ValueError("dO 的形状必须是 (B, Hq, Sq, D)")

    groups = hq // hkv
    q = Q.double()
    k = K.double()
    v = V.double()
    p = P.double()
    do = dO.double()

    v_h = v.repeat_interleave(groups, dim=1)          # (B,Hq,Sk,D)
    k_h = k.repeat_interleave(groups, dim=1)

    dP = do @ v_h.transpose(-1, -2)                   # (B,Hq,Sq,Sk)
    dS = (dP - (dP * p).sum(dim=-1, keepdim=True)) * p * float(scale)

    dQ = dS @ k_h
    dS_g = dS.view(b, hkv, groups, sq, sk)
    q_g = q.view(b, hkv, groups, sq, d)
    p_g = p.view(b, hkv, groups, sq, sk)
    do_g = do.view(b, hkv, groups, sq, d)

    dK = torch.einsum('bhgik,bhgid->bhkd', dS_g, q_g)
    dV = torch.einsum('bhgik,bhgid->bhkd', p_g, do_g)

    return dQ.to(Q.dtype), dK.to(K.dtype), dV.to(V.dtype)
