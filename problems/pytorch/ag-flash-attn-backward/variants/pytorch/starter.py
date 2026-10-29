import torch


def solve(Q, K, V, O, LSE, dO, scale=1.0, mask=None):
    """FlashAttention 反向：只给前向的 O 与 LSE（不存 n×n 概率矩阵），求 dQ/dK/dV。

    前向：S = (Q @ Kᵀ)·scale + mask，P = softmax(S)，O = P @ V，
    LSE[b,h,i] = log Σ_j exp(S[b,h,i,j])（已含 mask，形状 (B, H, Sq)）。

    入参：
    - Q: (B, H, Sq, D)；K/V: (B, H, Sk, D)；O: (B, H, Sq, D)；dO: 与 O 同形。
    - LSE: (B, H, Sq)；scale: float；mask: None 或可广播到 (B, H, Sq, Sk) 的加性掩码
      （0 表示可见、−inf 表示屏蔽）。任何形状不符都抛 ValueError。

    口径（与朴素实现逐位一致，但不需要保存 P）：
    - 逐块重算概率：P = exp(S + mask − LSE)，**减 LSE 而不是减 max**，这样 exp 结果
      直接就是归一化概率，掩码位因为 S = −inf 自动得 0，也就不会污染 dQ/dK。
    - dV = Pᵀ @ dO；dP = dO @ Vᵀ；D = rowsum(dO ∘ O)（每行一个标量）；
      dS = P ∘ (dP − D) · scale；dQ = dS @ K；dK = dSᵀ @ Q。
      D 这一项来自「重算时没有保存 P 的归一化」，漏掉它等价于忘了 softmax 的减均值。
    - 因为 P 由 S + mask − LSE 重建，把 LSE 整体加常数 c 的同时给 S（或 mask）加 c
      不改变结果；单独加 LSE 会让 P 缩放 e^(−c)，梯度也随之缩放——可用这条自检验证实现。
    - 内部 float64 计算，返回 dtype 与 Q/K/V 各自一致。
    """
    # TODO
    raise NotImplementedError
