def solve(x, W0, b0, boft_R, boft_s):
    """BOFT：蝴蝶分解的多级分块正交适配器（对齐 peft `tuners/boft/layer.py` 的
    `cayley_batch` + `get_delta_weight` + `forward`）。

    形状约定
        x       : (..., d_in)，任意前导维度
        W0      : (d_out, d_in) 基座权重
        b0      : (d_out,) 或 None
        boft_R  : (F, D, H, H) —— F 个蝴蝶级、每级 D 个 H×H 块
        boft_s  : (d_out, 1) —— 逐输出行的缩放

    合法条件：H 是正偶数；d_in = D * H（即 D = d_in // H 与 boft_R 的第 1 维一致）；
    F >= 1 且 D 能被 2**(F-1) 整除；boft_R 前两维必须是 (F, D)，boft_s 必须是 (d_out, 1)。

    1) 逐块 Cayley 变换（每级每块独立）
           S = (R - R.T) / 2                       # 强制反对称
           Q = (I + S)^(-1) @ (I - S)              # 右解：solve(I+S, I-S, left=False)
       Q 是 H×H 正交阵且 det = 1。注意 R 不必反对称，只有它的反对称部分起作用。

    2) 每级的分块对角阵
           B_i = blockdiag(Q_i[0], Q_i[1], ..., Q_i[D-1])      # (d_in, d_in)

    3) 每级的置换阵 P_i（蝴蝶重排）
       把长度 d_in 的下标轴切成 D // 2**i 段，每段长 L = H * 2**i；
       段内以 r = H // 2 个连续元素为一组（每段共 2**(i+1) 组），
       重排为「偶数号组按序在前，再接奇数号组按序」，组内元素顺序不变。
       第 0 级只有 0、1 两组，因此 P_0 恒为单位阵。
       置换向量 perm 满足「新位置 k 取原位置 perm[k]」，
       置换阵按 P[k, perm[k]] = 1 构造（peft 的 perm2mat）。

    4) 每级的蝴蝶块与级间连乘
           M_i = P_i @ B_i @ P_i.T
           M   = M_(F-1) @ ... @ M_1 @ M_0        # 级号从小到大依次左乘

    5) 权重先旋转、再缩放，最后前向
           W' = (M @ W0.T).T * boft_s             # 等价于 (W0 @ M.T) 逐输出行乘 s
           y  = x @ W'.T + b0                     # b0 为 None 时不加偏置

    返回 (y, M)：y 与 x 的前导维度一致、最后一维为 d_out；M 是 (d_in, d_in) 的正交阵。
    内部全部用 float64 计算，返回的 y 与 M 都是 float32。
    非法输入（形状不符、整除关系不成立、H 非偶数、非浮点、NaN/Inf 等）抛出 ValueError。
    禁止调用现成 PEFT 库。"""
    raise NotImplementedError
