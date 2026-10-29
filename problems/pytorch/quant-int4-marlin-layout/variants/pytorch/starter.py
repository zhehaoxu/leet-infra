import torch


def solve(qweight, qzeros, scales, perm, K, N):
    """4-bit 权重的 tile 交织布局：解码 nibble 并反量化回逻辑列顺序。

    qweight : int32 (K//16, N//16, 16, 2)
              维度依次是「tile 行, tile 列, tile 内行, tile 内半字」。一个 tile 是 16 行 × 16 列；
              第 t 个 tile 行覆盖原始行 16t..16t+15，第 g 个 tile 列覆盖存储列 16g..16g+15。
              tile 内第 r 行、半字 h（0 或 1）的一个 int32 打包该行第 8h..8h+7 个 tile 内列：
              从低 4 位到高 4 位，第 b 个 nibble（b = 0..7）对应 tile 内列 j = 8h + ORDER[b]，
              ORDER = [0, 2, 4, 6, 1, 3, 5, 7]。每个 nibble 是一个无符号 4-bit 码 0..15。
    qzeros : int32 (N//16, 2)
              与 qweight 同一套 nibble 规则，只是没有行维：第 g 个 16 列组、半字 h 的 int32 中，
              第 b 个 nibble 是「存储列」c = 16g + 8h + ORDER[b] 的 zero-point（0..15）。
    scales : float32 (N,)，按「逻辑列」给出反量化 scale。
    perm   : list[int]，长度 N，perm[c] 是存储列 c 对应的逻辑列号（0..N-1 的一个排列）。
    K, N   : 原始权重行数、列数，均为 16 的正整数倍。

    返回 W：float32 (K, N)，逻辑列顺序，满足
        W[k, perm[c]] = (code(k, c) - zero(c)) * scales[perm[c]]，
    其中 code(k, c) 是 qweight 中第 k 行、存储列 c 的 nibble。

    K % 16 != 0 或 N % 16 != 0 时抛 ValueError。
    """
    # TODO: 按 tile / 半字 / nibble 槽位解码，再按 perm 散射到逻辑列并反量化
    raise NotImplementedError
