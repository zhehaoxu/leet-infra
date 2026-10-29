import torch


def solve(rewards):
    """RLOO 优势：A[b,i] = r[b,i] - mean_{j != i} r[b,j]。

    入参：
    - rewards: (B,G) float32/float64，B >= 1 个 prompt，每个 prompt 采样
      G >= 2 条回答的标量奖励（G 是组大小）。

    口径：
    - 逐个 (b,i) 用**同组其余 G-1 个**奖励的均值当基线：
        A[b,i] = r[b,i] - (Σ_{j} r[b,j] - r[b,i]) / (G-1)；
      分母必须是 G-1 而不是 G（留一），也不是 std（不做方差归一）。
    - 返回 (B,G) **float64**，与入参同形状；逐组求和恒为 0
      （Σ_i A[b,i] = 0），这是留一基线的直接推论。
    - G = 1 时基线无定义，必须抛 ValueError，不许退化成 A = 0。
    - 不调用随机数；CPU、确定性。

    非法输入抛 ValueError：非张量 / 非二维 / 非浮点 / 含 NaN·Inf /
    B < 1 / G < 2。
    """
    raise NotImplementedError
