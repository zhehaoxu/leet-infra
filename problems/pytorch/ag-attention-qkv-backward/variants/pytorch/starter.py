import torch


def solve(Q, K, V, P, dO, scale=1.0):
    """缩放点积注意力（含 GQA）对 Q/K/V 的反向：返回 (dQ, dK, dV)。

    前向：scores = (Q @ Kᵀ) · scale + mask，P = softmax(scores, 末轴)，O = P @ V_head。
    K/V 的头数是 Hkv，Q 的头数是 Hq = groups · Hkv，第 h 个 Q 头用 K/V 的第 h // groups 头。

    入参：
    - Q: (B, Hq, Sq, D)；K: (B, Hkv, Sk, D)；V: (B, Hkv, Sk, D)；D 为 head_dim。
    - P: (B, Hq, Sq, Sk) 前向已算好的注意力概率（掩码位为 0）。
    - dO: (B, Hq, Sq, D) 上游对输出的梯度。
    - scale: float，默认 1.0（真实模型里是 1/sqrt(D)，这里作为显式入参便于手算）。

    口径：
    - dP = dO @ V_headᵀ；对概率的 softmax 反向 d(scores) = P ∘ (dP − Σ_k P_k dP_k)，
      再乘 scale 才是对 Q@Kᵀ 的梯度。**掩码位 P=0 ⇒ 该处梯度必为 0**，无需额外掩码。
    - dQ = dS @ K_head，形状同 Q；dK/dV 要把同组 groups 个 Q 头的贡献**求和**，
      形状是 (B, Hkv, ·, ·)，不是 (B, Hq, ·, ·)。
    - dV = Σ_groups P_headᵀ @ dO。
    - Hq % Hkv != 0、形状不一致、K 与 V 的头数不等都抛 ValueError；
      dtype 与 Q 一致（内部 float64 累计）。
    """
    # TODO
    raise NotImplementedError
