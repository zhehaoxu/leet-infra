import torch


def solve(x, u, variant='sr'):
    """把 float32 按 u 里的均匀噪声随机舍入到 bfloat16 网格，支持无偏 SR 与 fwd/bwd 两种偏置。

    随机数由入参 u 提供（真实 kernel 里是 philox/threefry 产生的噪声流），所以本题完全确定性。
    不许调用 x.to(torch.bfloat16)，也不许用 torch.round 做最终取舍。

    关键结构：bf16 就是 float32 的高 16 位，所以**格点在「高 16 位整数」上是均匀的**。
    取 x 的 32 位模式（按无符号整数读）u32：
        lo = (u32 >> 16) << 16          # 朝零方向的 bf16 格点
        hi = (u32 >> 16) + 1 再 << 16    # 幅值大一号的相邻格点
        r  = (u32 & 0xFFFF) / 65536.0   # x 在 [lo, hi) 内的相对位置，就是 ulp 分数
    （r 对正负数都描述「幅值」方向，因为符号位单独占最高位。）

    三种 variant 取 hi 的概率 p：
    - 'sr'  （无偏）:            p = r
    - 'fwd' （偏向零，激活用）:   r <= 0.5 时 p = 0；否则 p = 2r - 1
    - 'bwd' （远离零，梯度用）:   r >= 0.5 时 p = 1；否则 p = 2r
    判定规则是 `u < p`（严格小于）则取 hi，否则取 lo。
    于是 E[|SR(x)|] = |x|（sr）、<= |x|（fwd）、>= |x|（bwd）：
    fwd 把小残差直接抹成 0，累加会系统性下偏；bwd 保证梯度累加不会因为截断而缩小。

    特殊值（都不看 u）：
    - r == 0（已经在 bf16 格点上，含 ±0、±inf）：原值返回，±0 保留符号。
    - NaN：输出仍是 NaN（符号位不保证保留）。
    - |x| 超过 bf16 最大有限值时，取 hi 会自然溢出成 ±inf（不做饱和）。

    入参校验：x 与 u 都是 float32 且形状完全相同（不广播）；u 的每个元素必须在 [0, 1) 内，
    否则抛 ValueError；variant 不在三种之内抛 ValueError；非 Tensor 输入抛 ValueError。
    返回 float32 张量，形状同 x。
    """
    # TODO
    raise NotImplementedError
