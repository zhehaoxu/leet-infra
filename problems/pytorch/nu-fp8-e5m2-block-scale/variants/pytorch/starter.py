import torch


def solve(x, block=32):
    """把张量按连续的 block 个元素分块，每块共享一个 2 的幂尺度量化到 FP8 E5M2 再反量化。

    不许调用 x.to(torch.float8_e5m2)/torch.float8_e4m3fn，也不许借 torch 的 autocast——
    考点是自己把 E5M2 网格和块尺度算出来。

    E5M2 可表示的幅值（1 位符号 + 5 位指数（bias 15）+ 2 位尾数）：
    - 正规数：(1 + M/4) * 2^e，e = -14..15，M = 0..3 → 最大有限值 57344 = 1.75 * 2^15；
    - 次正规数：M * 2^-16，M = 0..3（步长 2^-16 = 1.52587890625e-05）；
    - 每个 2^e 区间里格点步长是 2^(e-2)，也就是每倍频只有 4 个格点。

    流程（按 row-major 展平后每 block 个连续元素为一块，最后一块可以不足 block 个）：
    1. m = 块内 |x| 的最大值。
    2. scale = 2^e，e 是**满足 m <= 57344 * 2^e 的最小整数**（也就是「最小的、不会让块内
       最大元素溢出的 2 的幂尺度」），并把 e 夹到 [-127, 127]。m == 0 时 e = 0（scale = 1.0）。
       注意：`math.ceil(math.log2(m / 57344))` 在 m/57344 正好是 2 的幂时会因为浮点误差多加 1，
       得到偏大一倍的 scale——这是本题最容易翻车的地方。
    3. 对块内每个元素：t = x / scale，把 t 舍入到最近的 E5M2 格点（**ties-to-even**，
       即恰好落在两格点中间时取尾数域 M 为偶数的那个），再乘回 scale。
       舍入结果里的 0 一律写成 +0.0。

    因为 scale 保证 |t| <= 57344，量化不会溢出到 inf；所以输出里不会出现 inf。

    特殊值：
    - 块内 absmax 是 inf 或 nan（含块里有 nan）→ 该块所有输出都是 nan。
    - 全 0 块 → 输出全 0。

    入参 x 是 float32 或 float64 张量（任意形状），输出同形状同 dtype。
    block 是 >= 1 的整数，否则抛 ValueError；非浮点张量、非 Tensor 也抛 ValueError。
    """
    # TODO
    raise NotImplementedError
