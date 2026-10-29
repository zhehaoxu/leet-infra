"""参考答案：用 LSE 把 S 平移成真实概率，再走标准 softmax + matmul 反向。

flash 反向的全部技巧都在于「不存 P」：
P = exp(S + mask − LSE) 里的 LSE 是前向在线 softmax 顺手记下的行统计量，减掉它之后 exp
就是归一化概率本身（不需要再除和），掩码位自然为 0。
D = rowsum(dO ∘ O) 是把 softmax 反向里的 Σ_k P_k dP_k 改写成不需要 P 的形式：
Σ_j P_j dP_j = Σ_j P_j (dO·V_j) = dO · (Σ_j P_j V_j) = dO · O_i，
于是这一项只需 O 与 dO 的点积——省掉了 n×n 的中间量。
"""

import torch


def solve(Q, K, V, O, LSE, dO, scale=1.0, mask=None):
    if Q.dim() != 4 or K.dim() != 4 or V.dim() != 4:
        raise ValueError("Q/K/V 必须是 (B, H, S, D) 四维张量")
    b, h, sq, d = Q.shape
    bk, hk, sk, dk = K.shape
    if tuple(V.shape) != (b, h, sk, d):
        raise ValueError("V 必须与 K 同形且 batch/head 与 Q 一致")
    if (bk, hk, dk) != (b, h, d):
        raise ValueError("K 的 batch/head/head_dim 必须与 Q 一致")
    if tuple(O.shape) != (b, h, sq, d) or tuple(dO.shape) != (b, h, sq, d):
        raise ValueError("O 与 dO 必须是 (B, H, Sq, D)")
    if tuple(LSE.shape) != (b, h, sq):
        raise ValueError("LSE 必须是 (B, H, Sq)")

    m = None
    if mask is not None:
        if mask.dim() > 4:
            raise ValueError("mask 必须可广播到 (B, H, Sq, Sk)")
        try:
            m = mask.double().expand(b, h, sq, sk)
        except RuntimeError:
            raise ValueError("mask 必须可广播到 (B, H, Sq, Sk)")

    q = Q.double()
    k = K.double()
    v = V.double()
    o = O.double()
    do = dO.double()
    lse = LSE.double()

    S = (q @ k.transpose(-1, -2)) * float(scale)
    if m is not None:
        S = S + m
    P = torch.exp(S - lse.unsqueeze(-1))          # 减 LSE ⇒ 直接是归一化概率

    dV = torch.einsum('bhij,bhid->bhjd', P, do)
    dP = do @ v.transpose(-1, -2)
    D = (do * o).sum(dim=-1, keepdim=True)         # = Σ_j P_j dP_j，靠 O 免存 P
    dS = P * (dP - D) * float(scale)

    dQ = dS @ k
    dK = dS.transpose(-1, -2) @ q
    return dQ.to(Q.dtype), dK.to(K.dtype), dV.to(V.dtype)
